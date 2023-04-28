#include <gflags/gflags.h>
#include <glog/logging.h>

#include "distopt/equidistant_distortion.hpp"
#include "distopt/frame.hpp"
#include "distopt/pinhole_camera.hpp"
#include "distopt/rad_distortion.hpp"
#include "distopt/radtan_distortion.hpp"
#include "distopt/simple_pinhole_camera.hpp"
#include "distopt/utils.hpp"

DEFINE_double(alpha_intrinsics, 1.0e5,
              "The consensus weight parameter for the intrinsics");
DEFINE_double(alpha_distortion, 1.0e5,
              "The consensus weight parameter for the distortion");
DEFINE_double(alpha_rotation, 1.0e5,
              "The consensus weight parameter for the rotation");
DEFINE_double(alpha_translation, 1.0e5,
              "The consensus weight parameter for the translation");
DEFINE_double(asynch_nu, 1.0,
              "The weighting factor for the asynchronous dual update");

namespace dba {

Frame::Frame(const uint64_t id, const uint64_t graph_id,
             const Eigen::Vector3d& p_W_C, const Eigen::Quaterniond& q_W_C,
             const Eigen::VectorXd& intrinsics,
             const Eigen::VectorXd& dist_coeffs, const Camera::Type& cam_type,
             const Distortion::Type& dist_type)
    : id_(id),
      graph_id_(graph_id),
      p_W_C_(p_W_C),
      q_W_C_(q_W_C),
      q_ref_(q_W_C),
      intrinsics_(intrinsics),
      dist_coeffs_(dist_coeffs),
      is_valid_(true),
      cam_type_(cam_type),
      dist_type_(dist_type) {
  if (cam_type == Camera::Type::kPinholeSimple) {
    DistortionUniquePtr distortion;
    switch (dist_type) {
      case Distortion::Type::kRadDist:
        distortion.reset(new RadDistortion(dist_coeffs));
        break;
      case Distortion::Type::kRadTanDist:
        distortion.reset(new RadTanDistortion(dist_coeffs));
        break;
      case Distortion::Type::kEquiDist:
        distortion.reset(new EquidistantDistortion(dist_coeffs));
        break;
    }
    camera_ptr_.reset(new SimplePinholeCamera(intrinsics, distortion));
  } else if (cam_type == Camera::Type::kPinhole) {
    DistortionUniquePtr distortion;
    switch (dist_type) {
      case Distortion::Type::kRadDist:
        distortion.reset(new RadDistortion(dist_coeffs));
        break;
      case Distortion::Type::kRadTanDist:
        distortion.reset(new RadTanDistortion(dist_coeffs));
        break;
      case Distortion::Type::kEquiDist:
        distortion.reset(new EquidistantDistortion(dist_coeffs));
        break;
    }
    camera_ptr_.reset(new PinholeCamera(intrinsics, distortion));
  }
  central_dual_.setId(id_);
  central_dual_.fill(0.0);
}

Frame::~Frame() {}

auto Frame::setNeighborInvalid(const uint64_t neigh_id) -> void {
  z_invalid_.insert(neigh_id);
}

auto Frame::isNeighborInvalid(const uint64_t neigh_id) const -> bool {
  return z_invalid_.count(neigh_id);
}

auto Frame::setDualData(const uint64_t neigh_id, const FrameDual& dual)
    -> void {
  if (!z_.count(neigh_id)) z_[neigh_id] = FrameDual(id_);
  auto& z = z_[neigh_id];
  for (size_t i = 0; i < z.getSize(); ++i) z[i] = dual[i];
}

auto Frame::getDualData(const uint64_t neigh_id, FrameDual& dual) -> bool {
  if (!z_.count(neigh_id)) return false;
  const auto& z = z_[neigh_id];
  for (size_t i = 0; i < z.getSize(); ++i) dual[i] = z[i];
  return true;
}

auto Frame::updateDualVariables() -> bool {
  if (z_hat_.empty()) return false;

  // Compute the state vector
  FrameDual x_lambda;
  Eigen::Map<Eigen::Vector3d> dual_trans(x_lambda.getPosition());
  dual_trans = lambda_trans_ * p_W_C_;
  Eigen::Vector3d delta_q;
  utils::rotmath::Minus(q_W_C_, q_ref_, &delta_q);
  Eigen::Map<Eigen::Vector3d> dual_rot(x_lambda.getRotation());
  dual_rot = lambda_rot_ * delta_q;
  Eigen::Map<Eigen::Matrix<double, kNumIntrinsicParams, 1>> dual_intr(
      x_lambda.getIntrisincs());
  dual_intr = lambda_intr_ * intrinsics_;
  Eigen::Map<Eigen::Matrix<double, kNumDistortionParams, 1>> dual_dist(
      x_lambda.getDistortion());
  dual_dist = lambda_dist_ * dist_coeffs_;
  // Equations (8) (9) (10)
  for (const auto& [id, z_hat_i] : z_hat_) {
    CHECK(z_.count(id));
    auto& z_i = z_[id];
    for (size_t i = 0; i < x_lambda.getSize(); ++i)
      z_i[i] -= FLAGS_asynch_nu * ((z_i[i] + z_hat_i[i]) * 0.5 + x_lambda[i]);
  }
  return true;
}

auto Frame::setCommDual(const uint64_t neigh_id, const FrameDual& dual)
    -> void {
  if (!z_hat_.count(neigh_id)) z_hat_[neigh_id] = FrameDual(id_);
  auto& z_hat = z_hat_[neigh_id];
  for (size_t i = 0; i < z_hat.getSize(); ++i) z_hat[i] = dual[i];
}

auto Frame::getCommDual(const uint64_t neigh_id, FrameDual& dual) -> bool {
  if (!z_hat_.count(neigh_id)) return false;
  const auto& z_hat = z_hat_[neigh_id];
  for (size_t i = 0; i < z_hat.getSize(); ++i) dual[i] = z_hat[i];
  return true;
}

auto Frame::setCentralDual(const FrameDual& dual) -> void {
  for (size_t i = 0; i < dual.getSize(); ++i) central_dual_[i] = dual[i];
}

auto Frame::getCentralDual(FrameDual& dual) -> bool {
  if (dual[0] > 1e12) return false;
  dual.setId(id_);
  for (size_t i = 0; i < dual.getSize(); ++i) dual[i] = central_dual_[i];
  return true;
}

auto Frame::addObservation(const Observation& obs) -> bool {
  if (observations_.count(obs.mp_id)) return false;
  observations_[obs.mp_id] = obs;
  return true;
}

auto Frame::getAllObservations() -> ObservationVector {
  ObservationVector observations;
  observations.reserve(observations_.size());
  for (const auto& [id, obs] : observations_) observations.push_back(obs);
  return observations;
}

auto Frame::getCamera() const -> CameraSharedPtr { return camera_ptr_; }

}  // namespace dba
