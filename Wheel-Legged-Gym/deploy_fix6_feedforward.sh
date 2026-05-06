#!/bin/bash
# Fix6: Feedforward lean + increased action_scale_theta + reduced wheel damping
# Deploy to server and start training

SERVER="root@192.168.0.201"
REMOTE_DIR="/root/Wheel_Legged_Gym_2/wlr_ws/Wheel-Legged-Gym/wheel_legged_gym/envs/wheel_legged_vmc_flat"

echo "=== Deploying Fix6 files ==="
scp wheel_legged_gym/envs/wheel_legged_vmc_flat/wheel_legged_residual_flat.py \
    wheel_legged_gym/envs/wheel_legged_vmc_flat/wheel_legged_residual_flat_config.py \
    "${SERVER}:${REMOTE_DIR}/"

echo "=== Killing old training processes ==="
ssh "${SERVER}" "pkill -f train.py || true; sleep 2"

echo "=== Starting new training ==="
ssh "${SERVER}" "cd /root/Wheel_Legged_Gym_2/wlr_ws/Wheel-Legged-Gym && nohup python -u scripts/train.py --task wheel_legged_residual_flat --headless --resume > logs/fix6_ff_lean.log 2>&1 &"

echo "=== Done. Monitor with: ==="
echo "ssh ${SERVER} 'tail -f /root/Wheel_Legged_Gym_2/wlr_ws/Wheel-Legged-Gym/logs/fix6_ff_lean.log'"
