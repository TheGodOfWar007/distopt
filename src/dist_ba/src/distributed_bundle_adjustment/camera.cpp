#include "distributed_bundle_adjustment/camera.hpp"

namespace dba {

Camera::Camera(const Eigen::VectorXd& intrinsics,
               DistortionUniquePtr& distortion, Type camera_type)
    : intrinsics_(intrinsics),
      camera_type_(camera_type),
      distortion_(std::move(distortion)) {}

Camera::Camera(const Camera& other) : distortion_(nullptr) {
  distortion_.reset(other.distortion_->clone());
}

Camera::~Camera() {}

auto Camera::projectPoint(
    const Eigen::Vector3d& point_in_C, Eigen::Vector2d* projection,
    Eigen::Matrix<double, 2, 3>* jacobian_wrt_point,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* jacobian_wrt_intrinsics,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* jacobian_wrt_dist) const
    -> ProjectionResult {
  return projectPointUsingExternalParameters(
      intrinsics_, distortion_->getParameters(), point_in_C, projection,
      jacobian_wrt_point, jacobian_wrt_intrinsics, jacobian_wrt_dist);
}

}  // namespace dba
