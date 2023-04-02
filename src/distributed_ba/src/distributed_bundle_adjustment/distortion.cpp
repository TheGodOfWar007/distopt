#include "distributed_bundle_adjustment/distortion.hpp"

namespace dba {

Distortion::Distortion(const Eigen::VectorXd& dist_coeffs, Type distortion_type)
    : dist_coeffs_(dist_coeffs), dist_type_(distortion_type) {}

Distortion::~Distortion() {}

auto Distortion::distortPoint(
    const Eigen::Vector2d& point_in, Eigen::Vector2d* point_out,
    Eigen::Matrix2d* jacobian_wrt_point,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* jacobian_wrt_dist) const -> void {
  distortPointUsingExternalParameters(dist_coeffs_, point_in, point_out,
                                      jacobian_wrt_point, jacobian_wrt_dist);
}

auto Distortion::getParameters() const -> const Eigen::VectorXd& {
  return dist_coeffs_;
}

}  // namespace dba
