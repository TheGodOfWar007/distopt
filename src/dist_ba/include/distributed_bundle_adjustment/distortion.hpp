#pragma once

#include <Eigen/Dense>
#include <memory>

#include "distributed_bundle_adjustment/common.hpp"

namespace dba {

class Distortion {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum class Type { kRadDist = 0, kRadTanDist = 1, kEquiDist = 2 };

 protected:
  Distortion() = delete;
  Distortion(const Eigen::VectorXd& dist_coeffs, Type distortion_type);
  // Copy constructor for clone operation.
  Distortion(const Distortion&) = default;
  void operator=(const Distortion&) = delete;

 public:
  virtual ~Distortion();

  virtual auto clone() const -> Distortion* = 0;

  virtual auto distortPoint(
      const Eigen::Vector2d& point_in, Eigen::Vector2d* point_out,
      Eigen::Matrix2d* jacobian_wrt_point,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* jacobian_wrt_dist) const
      -> void;

  virtual auto distortPointUsingExternalParameters(
      const Eigen::VectorXd& dist_coeffs, const Eigen::Vector2d& point_in,
      Eigen::Vector2d* point_out, Eigen::Matrix2d* jacobian_wrt_point,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* jacobian_wrt_dist) const
      -> void = 0;

  virtual auto getParameters() const -> const Eigen::VectorXd&;

 protected:
  Eigen::VectorXd dist_coeffs_;
  Type dist_type_;
};

using DistortionSharedPtr = std::shared_ptr<Distortion>;
using DistortionUniquePtr = std::unique_ptr<Distortion>;

}  // namespace dba
