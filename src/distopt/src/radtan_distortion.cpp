#include <glog/logging.h>

#include "distopt/radtan_distortion.hpp"

namespace dba {

RadTanDistortion::RadTanDistortion(const Eigen::VectorXd& dist_coeffs)
    : Base(dist_coeffs, Distortion::Type::kRadDist) {}

auto RadTanDistortion::distortPointUsingExternalParameters(
    const Eigen::VectorXd& dist_coeffs, const Eigen::Vector2d& point_in,
    Eigen::Vector2d* point_out, Eigen::Matrix2d* jacobian_wrt_point,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* jacobian_wrt_dist) const -> void {
  CHECK(point_out != nullptr);
  CHECK(dist_coeffs.size() == kNumParameters);
  const double x = point_in[0];
  const double y = point_in[1];
  const double& k1 = dist_coeffs[0];
  const double& k2 = dist_coeffs[1];
  const double& p1 = dist_coeffs[2];
  const double& p2 = dist_coeffs[3];
  const double& k3 = dist_coeffs[4];
  const double rad = point_in.norm();
  const double rad2 = rad * rad;
  const double rad4 = rad2 * rad2;
  const double rad6 = rad4 * rad2;
  const double mx2 = x * x;
  const double my2 = y * y;
  const double mxy = x * y;
  const double rad_fact = 1.0 + k1 * rad2 + k2 * rad4 + k3 * rad6;
  const double tan_fact_x = 2.0 * p1 * mxy + p2 * (rad2 + 2 * mx2);
  const double tan_fact_y = 2.0 * p1 * mxy + p2 * (rad2 + 2 * my2);

  (*point_out)[0] = x * rad_fact + tan_fact_x;
  (*point_out)[1] = y * rad_fact + tan_fact_y;
  if (jacobian_wrt_point != nullptr) {
    const double du_wrt_dx =
        k2 * rad4 + k3 * rad6 + 6.0 * p2 * x + 2.0 * p1 * y +
        x * (2.0 * k1 * x + 4.0 * k2 * x * rad2 + 6.0 * k3 * x * rad4) +
        k1 * rad2 + 1.0;
    const double du_wrt_dy =
        2.0 * p1 * x + 2.0 * p2 * y +
        x * (2.0 * k1 * y + 4.0 * k2 * y * rad2 + 6.0 * k3 * y * rad4);
    const double dv_wrt_dx =
        2.0 * p2 * x + 2.0 * p1 * y +
        y * (2.0 * k1 * x + 4.0 * k2 * x * rad2 + 6.0 * k3 * x * rad4);
    const double dv_wrt_dy =
        k2 * rad4 + k3 * rad6 + 2.0 * p1 * x + 6.0 * p2 * y +
        y * (2.0 * k1 * y + 4.0 * k2 * y * rad2 + 6.0 * k3 * y * rad4) +
        k1 * rad2 + 1.0;
    (*jacobian_wrt_point) << du_wrt_dx, du_wrt_dy, dv_wrt_dx, dv_wrt_dy;
  }
  if (jacobian_wrt_dist != nullptr) {
    jacobian_wrt_dist->resize(2, kNumParameters);
    const double du_wrt_k1 = x * rad2;
    const double du_wrt_k2 = x * rad4;
    const double du_wrt_p1 = 2.0 * mxy;
    const double du_wrt_p2 = 3.0 * mx2 + my2;
    const double du_wrt_k3 = x * rad6;
    const double dv_wrt_k1 = y * rad2;
    const double dv_wrt_k2 = y * rad4;
    const double dv_wrt_p1 = 2.0 * mxy;
    const double dv_wrt_p2 = 3.0 * my2 + mx2;
    const double dv_wrt_k3 = y * rad6;
    (*jacobian_wrt_dist) << du_wrt_k1, du_wrt_k2, du_wrt_p1, du_wrt_p2,
        du_wrt_k3, dv_wrt_k1, dv_wrt_k2, dv_wrt_p1, dv_wrt_p2, dv_wrt_k3;
  }
}

}  // namespace dba
