#pragma once

#include <Eigen/Dense>
#include <memory>

#include "distopt/camera.hpp"
#include "distopt/crpt_clone.hpp"
#include "distopt/distortion.hpp"

namespace dba {

class PinholeCamera : public Cloneable<Camera, PinholeCamera> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static constexpr int kNumParameters = 4;

 public:
  PinholeCamera(const Eigen::VectorXd& intrinsics,
                DistortionUniquePtr& distortion);

  // Copy constructor for clone operation.
  PinholeCamera(const PinholeCamera& other) = default;
  void operator=(const PinholeCamera&) = delete;

  virtual ~PinholeCamera(){};

 public:
  virtual auto projectPointUsingExternalParameters(
      const Eigen::VectorXd& camera_coeffs, const Eigen::VectorXd& dist_coeffs,
      const Eigen::Vector3d& point_in_C, Eigen::Vector2d* projection,
      Eigen::Matrix<double, 2, 3>* jacobian_wrt_point,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* jacobian_wrt_intrinsics,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* jacobian_wrt_dist) const
      -> ProjectionResult;
};

}  // namespace dba
