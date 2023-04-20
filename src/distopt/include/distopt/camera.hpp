#pragma once

#include <Eigen/Dense>
#include <memory>

#include "distopt/distortion.hpp"

namespace dba {

class Camera {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum class Type { kPinholeSimple = 0, kPinhole = 1 };

  enum class ProjectionResult {
    SUCCESSFUL,
    POINT_BEHIND_CAMERA,
    PROJECTION_INVALID
  };

 protected:
  Camera() = delete;
  Camera(const Eigen::VectorXd& intrinsics, DistortionUniquePtr& distortion,
         Type camera_type_);

  Camera(const Camera& other);

 public:
  virtual ~Camera();

  virtual auto clone() const -> Camera* = 0;

  virtual auto projectPoint(
      const Eigen::Vector3d& point_in_C, Eigen::Vector2d* projection,
      Eigen::Matrix<double, 2, 3>* jacobian_wrt_point,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* jacobian_wrt_intrinsics,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* jacobian_wrt_dist) const
      -> ProjectionResult;

  virtual auto projectPointUsingExternalParameters(
      const Eigen::VectorXd& camera_coeffs, const Eigen::VectorXd& dist_coeffs,
      const Eigen::Vector3d& point_in_C, Eigen::Vector2d* projection,
      Eigen::Matrix<double, 2, 3>* jacobian_wrt_point,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* jacobian_wrt_intrinsics,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* jacobian_wrt_dist) const
      -> ProjectionResult = 0;

 protected:
  Eigen::VectorXd intrinsics_;
  Type camera_type_;
  DistortionUniquePtr distortion_;
};

using CameraSharedPtr = std::shared_ptr<Camera>;
using CameraUniquePtr = std::shared_ptr<Camera>;

}  // namespace dba
