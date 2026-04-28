/*
 * FiveBarClosure — Gazebo system plugin
 *
 * Enforces five-bar linkage closure by applying high-stiffness spring-damper
 * forces between foot_front and foot_rear links.  Y-axis rotation is left
 * free (acts as the revolute "bearing" joint at point C).
 *
 * SDF usage (inside <model>):
 *   <plugin filename="FiveBarClosure"
 *           name="wlr_gazebo::FiveBarClosure">
 *     <front_link>left_foot_front</front_link>
 *     <rear_link>left_foot_rear</rear_link>
 *     <stiffness>5000</stiffness>
 *     <damping>50</damping>
 *     <rotational_stiffness>100</rotational_stiffness>
 *     <rotational_damping>5</rotational_damping>
 *   </plugin>
 */

#include <string>

#include <gz/math/Pose3.hh>
#include <gz/math/Quaternion.hh>
#include <gz/math/Vector3.hh>

#include <gz/plugin/Register.hh>

#include <gz/sim/System.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Util.hh>

namespace wlr_gazebo
{

class FiveBarClosure
    : public gz::sim::System,
      public gz::sim::ISystemConfigure,
      public gz::sim::ISystemPreUpdate
{
public:
  void Configure(
      const gz::sim::Entity &_entity,
      const std::shared_ptr<const sdf::Element> &_sdf,
      gz::sim::EntityComponentManager &_ecm,
      gz::sim::EventManager &) override
  {
    this->model = gz::sim::Model(_entity);

    if (_sdf->HasElement("front_link"))
      this->frontLinkName = _sdf->Get<std::string>("front_link");
    if (_sdf->HasElement("rear_link"))
      this->rearLinkName = _sdf->Get<std::string>("rear_link");

    if (_sdf->HasElement("stiffness"))
      this->kLin = _sdf->Get<double>("stiffness");
    if (_sdf->HasElement("damping"))
      this->cLin = _sdf->Get<double>("damping");
    if (_sdf->HasElement("rotational_stiffness"))
      this->kRot = _sdf->Get<double>("rotational_stiffness");
    if (_sdf->HasElement("rotational_damping"))
      this->cRot = _sdf->Get<double>("rotational_damping");

    gzmsg << "FiveBarClosure: front=" << this->frontLinkName
          << " rear=" << this->rearLinkName
          << " k=" << this->kLin << " c=" << this->cLin
          << " kR=" << this->kRot << " cR=" << this->cRot << std::endl;
  }

  void PreUpdate(
      const gz::sim::UpdateInfo &_info,
      gz::sim::EntityComponentManager &_ecm) override
  {
    if (_info.paused)
      return;

    // Lazy-resolve link entities (not available during Configure)
    if (!this->initialized)
    {
      auto frontEntity = this->model.LinkByName(_ecm, this->frontLinkName);
      auto rearEntity  = this->model.LinkByName(_ecm, this->rearLinkName);

      if (frontEntity == gz::sim::kNullEntity ||
          rearEntity  == gz::sim::kNullEntity)
      {
        return;  // links not spawned yet
      }

      this->frontLink = gz::sim::Link(frontEntity);
      this->rearLink  = gz::sim::Link(rearEntity);

      // Enable velocity reporting so we can compute damping
      this->frontLink.EnableVelocityChecks(_ecm, true);
      this->rearLink.EnableVelocityChecks(_ecm, true);

      this->initialized = true;
      this->warmupSteps = 100;  // skip first 100 steps (0.1s at 1kHz)
      gzmsg << "FiveBarClosure: resolved " << this->frontLinkName
            << " and " << this->rearLinkName << std::endl;
      return;  // skip this step
    }

    // Warm-up: let DART settle before applying constraint forces
    if (this->warmupSteps > 0)
    {
      --this->warmupSteps;
      return;
    }

    // Get world poses
    auto frontPoseOpt = this->frontLink.WorldPose(_ecm);
    auto rearPoseOpt  = this->rearLink.WorldPose(_ecm);
    if (!frontPoseOpt || !rearPoseOpt)
      return;

    const auto &pF = *frontPoseOpt;
    const auto &pR = *rearPoseOpt;

    // ---- Translational spring-damper ----
    // Error: vector from rear to front (we want them coincident)
    gz::math::Vector3d posErr = pF.Pos() - pR.Pos();

    // Velocity damping
    gz::math::Vector3d velErr(0, 0, 0);
    auto vF = this->frontLink.WorldLinearVelocity(_ecm);
    auto vR = this->rearLink.WorldLinearVelocity(_ecm);
    if (vF && vR)
      velErr = *vF - *vR;

    // Spring-damper force (applied to rear, reaction on front)
    gz::math::Vector3d force = this->kLin * posErr + this->cLin * velErr;

    // ---- Rotational spring-damper (X and Z axes only, Y is free) ----
    // Relative rotation: qRel = qF * qR^(-1)
    // Small-angle: the rotation vector components give the angular error
    gz::math::Quaterniond qRel = pF.Rot() * pR.Rot().Inverse();
    gz::math::Vector3d rotVec(qRel.Roll(), qRel.Pitch(), qRel.Yaw());

    // Zero out Y component — that's the free revolute DOF
    rotVec.Y(0.0);

    // Angular velocity damping (X and Z only)
    gz::math::Vector3d angVelErr(0, 0, 0);
    auto wF = this->frontLink.WorldAngularVelocity(_ecm);
    auto wR = this->rearLink.WorldAngularVelocity(_ecm);
    if (wF && wR)
    {
      angVelErr = *wF - *wR;
      angVelErr.Y(0.0);  // Y is free
    }

    gz::math::Vector3d torque = this->kRot * rotVec + this->cRot * angVelErr;

    // Clamp forces to prevent numerical explosion
    double fMag = force.Length();
    if (fMag > 200.0)
      force *= 200.0 / fMag;
    double tMag = torque.Length();
    if (tMag > 20.0)
      torque *= 20.0 / tMag;

    // Apply equal-and-opposite wrenches
    this->rearLink.AddWorldWrench(_ecm, force, torque);
    this->frontLink.AddWorldWrench(_ecm, -force, -torque);
  }

private:
  gz::sim::Model model{gz::sim::kNullEntity};
  gz::sim::Link frontLink{gz::sim::kNullEntity};
  gz::sim::Link rearLink{gz::sim::kNullEntity};

  std::string frontLinkName{"foot_front"};
  std::string rearLinkName{"foot_rear"};

  double kLin = 20000.0;  // N/m
  double cLin = 200.0;    // Ns/m
  double kRot = 200.0;    // Nm/rad
  double cRot = 10.0;     // Nms/rad

  bool initialized = false;
  int warmupSteps = 0;
};

}  // namespace wlr_gazebo

GZ_ADD_PLUGIN(
    wlr_gazebo::FiveBarClosure,
    gz::sim::System,
    gz::sim::ISystemConfigure,
    gz::sim::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(
    wlr_gazebo::FiveBarClosure,
    "wlr_gazebo::FiveBarClosure")
