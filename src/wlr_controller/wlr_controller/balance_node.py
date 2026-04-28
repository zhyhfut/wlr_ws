"""
WLR Balance Controller Node

Main control loop implementing VMC + LQR balance control for the
dual wheeled-legged robot, ported from the STM32 firmware.

Control loop runs at 500 Hz:
1. Read joint states + IMU
2. Compute virtual leg (L0, phi0) via VMC forward kinematics
3. Extract 6 state variables for LQR
4. Gain-scheduled LQR → virtual torques (T, Tp)
5. Leg length PID → virtual force F
6. Jacobian transpose → physical joint torques
7. Publish effort commands

Subscribes:
  /joint_states (sensor_msgs/JointState)
  /imu/data (sensor_msgs/Imu)
  /cmd_vel (geometry_msgs/Twist)
  /cmd_height (std_msgs/Float64)

Publishes:
  /effort_controller/commands (std_msgs/Float64MultiArray)
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, Float64MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from . import vmc
from . import lqr_gains
from . import leg_jacobian

# Must match hip_init_front / hip_init_rear in leg.urdf.xacro
HIP_INIT_FRONT = -0.7
HIP_INIT_REAR = 0.7


class BalanceController(Node):
    def __init__(self):
        super().__init__('balance_controller')

        # Declare parameters
        self.declare_parameter('wheel_radius', 0.05)
        self.declare_parameter('leg_mass', 0.2423)
        self.declare_parameter('total_mass', 3.55)
        self.declare_parameter('lqr_tp_ratio', 0.4)
        self.declare_parameter('lqr_t_ratio', 0.8)
        self.declare_parameter('f_ratio', 1.0)
        self.declare_parameter('default_leg_length', 0.11)
        self.declare_parameter('leg_length_kp', 100.0)
        self.declare_parameter('leg_length_kd', 300.0)
        self.declare_parameter('max_leg_force', 30.0)
        self.declare_parameter('torque_limit', 5.0)
        self.declare_parameter('control_rate', 500.0)

        # Load parameters
        self.wheel_R = self.get_parameter('wheel_radius').value
        self.leg_mass = self.get_parameter('leg_mass').value
        self.total_mass = self.get_parameter('total_mass').value
        self.lqr_tp_ratio = self.get_parameter('lqr_tp_ratio').value
        self.lqr_t_ratio = self.get_parameter('lqr_t_ratio').value
        self.f_ratio = self.get_parameter('f_ratio').value
        self.target_leg_length = self.get_parameter('default_leg_length').value
        self.leg_kp = self.get_parameter('leg_length_kp').value
        self.leg_kd = self.get_parameter('leg_length_kd').value
        self.max_leg_force = self.get_parameter('max_leg_force').value
        self.torque_limit = self.get_parameter('torque_limit').value
        control_rate = self.get_parameter('control_rate').value

        # Gravity compensation force per leg
        self.gravity_comp = (self.total_mass - 0.34) * 9.81 / 2.0

        # Target state
        self.target_speed = 0.0
        self.target_position = 0.0
        self.target_yaw_rate = 0.0

        # Joint state storage
        self.joint_names = [
            'left_hip_front_joint', 'left_hip_rear_joint', 'left_wheel_joint',
            'right_hip_front_joint', 'right_hip_rear_joint', 'right_wheel_joint',
        ]
        self.joint_pos = np.zeros(6)
        self.joint_vel = np.zeros(6)
        self.joints_ready = False

        # IMU state
        self.body_pitch = 0.0
        self.body_dpitch = 0.0
        self.imu_ready = False

        self.log_counter = 0
        self.running_initialized = False
        self.wheel_offset = 0.0
        self.upright_ready = False  # wait for robot to be upright before balancing
        # Ramp: gradually transition leg target from current L0 to desired
        self.ramp_counter = 0
        self.ramp_duration = int(control_rate * 2.0)  # 2s ramp
        self.initial_leg_length = None

        # QoS for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.create_subscription(
            JointState, '/joint_states', self.joint_states_cb, 10)
        self.create_subscription(
            Imu, '/imu/data', self.imu_cb, sensor_qos)
        self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_cb, 10)
        self.create_subscription(
            Float64, '/cmd_height', self.cmd_height_cb, 10)

        # Publisher
        self.effort_pub = self.create_publisher(
            Float64MultiArray, '/effort_controller/commands', 10)

        # Control timer (uses sim time when use_sim_time=true)
        dt = 1.0 / control_rate
        self.timer = self.create_timer(dt, self.control_loop)

        self.get_logger().info(
            f'Balance controller started at {control_rate} Hz, '
            f'wheel_R={self.wheel_R}, leg_length={self.target_leg_length}')

    # ------------------------------------------------------------------ #
    #  Callbacks                                                          #
    # ------------------------------------------------------------------ #

    def joint_states_cb(self, msg: JointState):
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                if idx < len(msg.position):
                    self.joint_pos[i] = msg.position[idx]
                if idx < len(msg.velocity):
                    self.joint_vel[i] = msg.velocity[idx]
        self.joints_ready = True

    def imu_cb(self, msg: Imu):
        q = msg.orientation
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            self.body_pitch = math.copysign(math.pi / 2, sinp)
        else:
            self.body_pitch = math.asin(sinp)
        self.body_dpitch = msg.angular_velocity.y
        self.imu_ready = True

    def cmd_vel_cb(self, msg: Twist):
        self.target_speed = msg.linear.x
        self.target_yaw_rate = msg.angular.z

    def cmd_height_cb(self, msg: Float64):
        self.target_leg_length = max(0.08, min(0.17, msg.data))
        self.get_logger().info(f'Target leg length: {self.target_leg_length:.3f}')

    # ------------------------------------------------------------------ #
    #  Main control loop                                                  #
    # ------------------------------------------------------------------ #

    def control_loop(self):
        if not self.joints_ready or not self.imu_ready:
            return

        # Wait until the robot is roughly upright before engaging LQR.
        # During the ~5s spawner chain the robot may have tipped over;
        # DART resets velocities on "invalid pose" events so eventually
        # the robot settles on its wheels with small pitch.
        if not self.upright_ready:
            pitch = abs(self.body_pitch)
            dpitch = abs(self.body_dpitch)
            if pitch < 0.3 and dpitch < 3.0:
                self.upright_ready = True
                self.log_counter = 0
                self.get_logger().info(
                    f'Robot upright: pitch={math.degrees(self.body_pitch):.1f}deg '
                    f'dpitch={self.body_dpitch:.2f} — engaging balance control')
            else:
                self.log_counter += 1
                if self.log_counter % 500 == 0:
                    self.get_logger().info(
                        f'Waiting for upright: pitch={math.degrees(self.body_pitch):.1f}deg '
                        f'dpitch={self.body_dpitch:.2f}')
                self.publish_zero()
                return

        self._do_balance()

    def _do_balance(self):
        """Full VMC + LQR balance control."""
        HALF_PI = math.pi / 2.0

        # First-tick initialization: reset wheel reference to current position
        if not self.running_initialized:
            avg_wp = (self.joint_pos[2] + self.joint_pos[5]) / 2.0
            self.wheel_offset = avg_wp  # subtract this from all future readings
            self.target_position = 0.0
            self.running_initialized = True
            self.get_logger().info(
                f'RUNNING initialized: wheel_offset={self.wheel_offset:.3f}')

        # ---- URDF → VMC angles ----
        left_phi1  = -HALF_PI - self.joint_pos[0] - HIP_INIT_FRONT
        left_phi4  = -HALF_PI - self.joint_pos[1] - HIP_INIT_REAR
        left_dphi1 = -self.joint_vel[0]
        left_dphi4 = -self.joint_vel[1]

        right_phi1  = -HALF_PI - self.joint_pos[3] - HIP_INIT_FRONT
        right_phi4  = -HALF_PI - self.joint_pos[4] - HIP_INIT_REAR
        right_dphi1 = -self.joint_vel[3]
        right_dphi4 = -self.joint_vel[4]

        left_wheel_pos  = self.joint_pos[2] - self.wheel_offset
        left_wheel_vel  = self.joint_vel[2]
        right_wheel_pos = self.joint_pos[5] - self.wheel_offset
        right_wheel_vel = self.joint_vel[5]

        # ---- VMC Forward Kinematics ----
        try:
            left_L0,  left_phi0  = vmc.leg_pos(left_phi1, left_phi4)
            left_dL0, left_dAngle = vmc.leg_spd(
                left_phi1, left_phi4, left_dphi1, left_dphi4)
            right_L0,  right_phi0  = vmc.leg_pos(right_phi1, right_phi4)
            right_dL0, right_dAngle = vmc.leg_spd(
                right_phi1, right_phi4, right_dphi1, right_dphi4)
        except (ValueError, ZeroDivisionError):
            self.publish_zero()
            return

        if (math.isnan(left_L0) or math.isnan(right_L0)
                or left_L0 < 0.01 or right_L0 < 0.01):
            self.publish_zero()
            return

        avg_L0     = (left_L0 + right_L0) / 2.0
        avg_phi0   = (left_phi0 + right_phi0) / 2.0
        avg_dAngle = (left_dAngle + right_dAngle) / 2.0
        avg_dL0    = (left_dL0 + right_dL0) / 2.0

        # ---- State variables ----
        theta  = avg_phi0 - math.pi / 2.0 + self.body_pitch
        dTheta = avg_dAngle + self.body_dpitch

        avg_wheel_pos = (left_wheel_pos + right_wheel_pos) / 2.0
        avg_wheel_vel = (left_wheel_vel + right_wheel_vel) / 2.0
        x  = avg_wheel_pos * self.wheel_R
        dx = (avg_wheel_vel * self.wheel_R
              + avg_L0 * dTheta * math.cos(theta)
              + avg_dL0 * math.sin(theta))

        phi  = -self.body_pitch
        dPhi = -self.body_dpitch

        if abs(self.target_speed) > 0.01:
            self.target_position = x
        else:
            # Slowly move target toward current x to prevent chronic offset
            raw_err = x - self.target_position
            if abs(raw_err) > 0.1:
                self.target_position += raw_err * 0.005

        # Clamp position error for LQR stability
        x_err = max(-0.3, min(0.3, x - self.target_position))

        # ---- LQR ----
        K = lqr_gains.lqr_k(avg_L0)
        state_err = np.array([
            theta, dTheta,
            x_err, self.target_speed - dx,
            phi, dPhi,
        ])
        lqr_out = K @ state_err
        lqr_T  = lqr_out[0]
        lqr_Tp = lqr_out[1]

        # ---- Leg length ramp + PID ----
        # On first tick, record initial L0 and ramp from there to target
        if self.initial_leg_length is None:
            self.initial_leg_length = avg_L0
            self.get_logger().info(
                f'Initial L0={avg_L0:.4f}, ramping to {self.target_leg_length:.4f}')

        if self.ramp_counter < self.ramp_duration:
            alpha = self.ramp_counter / self.ramp_duration
            effective_target = (1.0 - alpha) * self.initial_leg_length + alpha * self.target_leg_length
            self.ramp_counter += 1
        else:
            effective_target = self.target_leg_length

        left_F  = self._leg_pid(effective_target, left_L0, left_dL0)
        right_F = self._leg_pid(effective_target, right_L0, right_dL0)

        # ---- VMC Jacobian → joint torques ----
        left_Tp  =  lqr_Tp * self.lqr_tp_ratio
        right_Tp = -lqr_Tp * self.lqr_tp_ratio

        try:
            left_T1, left_T2   = leg_jacobian.leg_conv(
                left_phi1, left_phi4, left_F, left_Tp)
            right_T1, right_T2 = leg_jacobian.leg_conv(
                right_phi1, right_phi4, right_F, right_Tp)
        except (ValueError, ZeroDivisionError):
            self.publish_zero()
            return

        wheel_T = -lqr_T * self.lqr_t_ratio

        # VMC→URDF frame: T_urdf = -T_vmc
        left_T1  = -left_T1
        left_T2  = -left_T2
        right_T1 = -right_T1
        right_T2 = -right_T2

        # Saturation
        TL = self.torque_limit
        left_T1  = self._sat(left_T1, TL)
        left_T2  = self._sat(left_T2, TL)
        right_T1 = self._sat(right_T1, TL)
        right_T2 = self._sat(right_T2, TL)
        wheel_T  = self._sat(wheel_T, TL)

        # Periodic log
        self.log_counter += 1
        if self.log_counter % 500 == 0:
            x_err_raw = x - self.target_position
            self.get_logger().info(
                f'L0={avg_L0:.4f} phi0={math.degrees(avg_phi0):.1f}deg '
                f'theta={theta:.3f} phi={phi:.3f} '
                f'x_err={x_err_raw:.3f} dx={dx:.3f} '
                f'lqr_T={lqr_T:.2f} lqr_Tp={lqr_Tp:.2f} '
                f'T1L={left_T1:.3f} T2L={left_T2:.3f} Tw={wheel_T:.3f}')

        # Safety cutoff — only if completely fallen (> 1.2 rad ≈ 69 deg)
        if abs(theta) > 1.2:
            self.publish_zero()
            return

        # Publish
        msg = Float64MultiArray()
        msg.data = [
            left_T1, left_T2,
            wheel_T + self.target_yaw_rate * 0.5,
            right_T1, right_T2,
            wheel_T - self.target_yaw_rate * 0.5,
        ]
        self.effort_pub.publish(msg)

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _leg_pid(self, target, current, dcurrent):
        err = target - current
        F = self.leg_kp * err - self.leg_kd * dcurrent + self.gravity_comp
        return self._sat(F, self.max_leg_force) * self.f_ratio

    @staticmethod
    def _sat(val, limit):
        return max(-limit, min(limit, val))

    def publish_zero(self):
        msg = Float64MultiArray()
        msg.data = [0.0] * 6
        self.effort_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BalanceController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
