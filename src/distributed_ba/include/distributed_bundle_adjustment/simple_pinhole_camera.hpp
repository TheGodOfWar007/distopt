#pragma once

#include <Eigen/Dense>
#include <memory>

#include "distributed_bundle_adjustment/camera.hpp"
#include "distributed_bundle_adjustment/crpt_clone.hpp"
#include "distributed_bundle_adjustment/distortion.hpp"

namespace dba {

class SimplePinholeCamera : public Cloneable<Camera, SimplePinholeCamera> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static constexpr int kNumParameters = 1;

 public:
  // SimplePinholeCamera();
  SimplePinholeCamera(const Eigen::VectorXd& intrinsics,
                      DistortionUniquePtr& distortion);

  // Copy constructor for clone operation.
  SimplePinholeCamera(const SimplePinholeCamera& other) = default;
  void operator=(const SimplePinholeCamera&) = delete;

  virtual ~SimplePinholeCamera(){};

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
