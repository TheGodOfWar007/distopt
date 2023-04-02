#include "distributed_bundle_adjustment/rad_distortion.hpp"

namespace dba {

RadDistortion::RadDistortion(const Eigen::VectorXd& dist_coeffs)
    : Base(dist_coeffs, Distortion::Type::kRadDist) {}

auto RadDistortion::distortPointUsingExternalParameters(
    const Eigen::VectorXd& dist_coeffs, const Eigen::Vector2d& point_in,
    Eigen::Vector2d* point_out, Eigen::Matrix2d* jacobian_wrt_point,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* jacobian_wrt_dist) const -> void {
  assert(point_out != nullptr);
  assert(dist_coeffs.size() == 2);
  const double norm = point_in.norm();
  const double sqr_norm = norm * norm;
  const double fact1 =
      (1.0 + dist_coeffs[0] * sqr_norm + dist_coeffs[1] * sqr_norm * sqr_norm);
  *point_out = fact1 * point_in;
  if (jacobian_wrt_point != nullptr) {
    const double fact2 = (2 * dist_coeffs[0] + 4 * sqr_norm * dist_coeffs[1]);
    *jacobian_wrt_point = fact2 * point_in * point_in.transpose() +
                          fact1 * Eigen::Matrix2d::Identity();
  }
  if (jacobian_wrt_dist != nullptr) {
    jacobian_wrt_dist->resize(2, kNumParameters);
    jacobian_wrt_dist->block<2, 1>(0, 0) = sqr_norm * point_in;
    jacobian_wrt_dist->block<2, 1>(0, 1) = sqr_norm * sqr_norm * point_in;
  }
}

}  // namespace dba
