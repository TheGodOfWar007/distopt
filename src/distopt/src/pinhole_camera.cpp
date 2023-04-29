#include <glog/logging.h>

#include "distopt/pinhole_camera.hpp"

namespace dba {

PinholeCamera::PinholeCamera(const Eigen::VectorXd& dist_coeffs,
                             DistortionUniquePtr& distortion)
    : Base(dist_coeffs, distortion, Camera::Type::kPinholeSimple) {}

auto PinholeCamera::projectPointUsingExternalParameters(
    const Eigen::VectorXd& intrinsics, const Eigen::VectorXd& dist_coeffs,
    const Eigen::Vector3d& point_in_C, Eigen::Vector2d* projection,
    Eigen::Matrix<double, 2, 3>* jacobian_wrt_point,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* jacobian_wrt_intrinsics,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* jacobian_wrt_dist) const
    -> ProjectionResult {
  CHECK(projection != nullptr);
  CHECK_EQ(intrinsics.size(), kNumParameters);
  if (point_in_C[2] <= 0) return Camera::ProjectionResult::POINT_BEHIND_CAMERA;
  if (point_in_C[2] <= 1e-7)
    return Camera::ProjectionResult::PROJECTION_INVALID;
  const Eigen::Vector2d proj = point_in_C.head<2>() / point_in_C[2];
  const double fx = intrinsics[0];
  const double fy = intrinsics[1];
  const double cx = intrinsics[2];
  const double cy = intrinsics[3];
  if (jacobian_wrt_point || jacobian_wrt_intrinsics || jacobian_wrt_dist) {
    Eigen::Matrix<double, 2, 2> jacobian_wrt_proj;
    Eigen::Matrix<double, 2, Eigen::Dynamic> jacobian_proj_wrt_dist;
    // For simplicity just evaluate the jacobian in the distortion, even though
    // maybe not all of them will be used
    Eigen::Vector2d dist_proj;
    distortion_->distortPointUsingExternalParameters(
        dist_coeffs, proj, &dist_proj, &jacobian_wrt_proj,
        &jacobian_proj_wrt_dist);
    (*projection)[0] = fx * dist_proj[0] + cx;
    (*projection)[1] = fy * dist_proj[1] + cy;
    if (jacobian_wrt_point != nullptr) {
      const double inv_Z = 1.0 / point_in_C[2];
      Eigen::Matrix<double, 2, 3> jacobian_p_wrt_pointC;
      jacobian_p_wrt_pointC << inv_Z, 0.0, -point_in_C[0] * inv_Z * inv_Z, 0.0,
          inv_Z, -point_in_C[1] * inv_Z * inv_Z;
      const double du_dx = fx * jacobian_wrt_proj(0, 0) * inv_Z;
      const double du_dy = fx * jacobian_wrt_proj(0, 1) * inv_Z;
      const double du_dz = -fx *
                           (jacobian_wrt_proj(0, 0) * point_in_C[0] +
                            jacobian_wrt_proj(0, 1) * point_in_C[1]) *
                           inv_Z * inv_Z;
      const double dv_dx = fy * jacobian_wrt_proj(1, 0) * inv_Z;
      const double dv_dy = fy * jacobian_wrt_proj(1, 1) * inv_Z;
      const double dv_dz = -fy *
                           (jacobian_wrt_proj(1, 0) * point_in_C[0] +
                            jacobian_wrt_proj(1, 1) * point_in_C[1]) *
                           inv_Z * inv_Z;
      (*jacobian_wrt_point) << du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz;
    }
    if (jacobian_wrt_intrinsics != nullptr) {
      jacobian_wrt_intrinsics->resize(2, kNumParameters);
      (*jacobian_wrt_intrinsics) << dist_proj[0], 0.0, 1.0, 0.0, 0.0,
          dist_proj[1], 0.0, 1.0;
    }
    if (jacobian_wrt_dist != nullptr) {
      *jacobian_wrt_dist = jacobian_proj_wrt_dist;
      jacobian_wrt_dist->row(0) *= fx;
      jacobian_wrt_dist->row(1) *= fy;
    }
  } else {
    Eigen::Vector2d dist_proj;
    distortion_->distortPointUsingExternalParameters(
        dist_coeffs, proj, &dist_proj, nullptr, nullptr);
    (*projection)[0] = fx * dist_proj[0] + cx;
    (*projection)[1] = fy * dist_proj[1] + cy;
  }
  return Camera::ProjectionResult::SUCCESSFUL;
}

}  // namespace dba
