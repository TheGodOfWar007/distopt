#pragma once

#include <Eigen/Dense>
#include <memory>

#include "distributed_bundle_adjustment/crpt_clone.hpp"
#include "distributed_bundle_adjustment/distortion.hpp"

namespace dba {

class RadTanDistortion : public Cloneable<Distortion, RadTanDistortion> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static constexpr int kNumParameters = 5;

 public:
  explicit RadTanDistortion(const Eigen::VectorXd& dist_coeffs);

  // Copy constructor for clone operation. (stolen from aslam_cv2)
  RadTanDistortion(const RadTanDistortion&) = default;
  void operator=(const RadTanDistortion&) = delete;

  virtual auto distortPointUsingExternalParameters(
      const Eigen::VectorXd& dist_coeffs, const Eigen::Vector2d& point_in,
      Eigen::Vector2d* point_out, Eigen::Matrix2d* jacobian_wrt_point,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* jacobian_wrt_dist) const
      -> void;
};

}  // namespace dba
