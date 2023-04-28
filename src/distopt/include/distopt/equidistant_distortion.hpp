#pragma once

#include <Eigen/Dense>
#include <memory>

#include "distopt/crpt_clone.hpp"
#include "distopt/distortion.hpp"

namespace dba {

class EquidistantDistortion
    : public Cloneable<Distortion, EquidistantDistortion> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static constexpr int kNumParameters = 4;

 public:
  explicit EquidistantDistortion(const Eigen::VectorXd& dist_coeffs);

  // Copy constructor for clone operation. (stolen from aslam_cv2)
  EquidistantDistortion(const EquidistantDistortion&) = default;
  void operator=(const EquidistantDistortion&) = delete;

  virtual auto distortPointUsingExternalParameters(
      const Eigen::VectorXd& dist_coeffs, const Eigen::Vector2d& point_in,
      Eigen::Vector2d* point_out, Eigen::Matrix2d* jacobian_wrt_point,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* jacobian_wrt_dist) const
      -> void;
};

}  // namespace dba
