#include <glog/logging.h>

#include "distributed_bundle_adjustment/equidistant_distortion.hpp"

namespace dba {

EquidistantDistortion::EquidistantDistortion(const Eigen::VectorXd& dist_coeffs)
    : Base(dist_coeffs, Distortion::Type::kRadDist) {}

auto EquidistantDistortion::distortPointUsingExternalParameters(
    const Eigen::VectorXd& dist_coeffs, const Eigen::Vector2d& point_in,
    Eigen::Vector2d* point_out, Eigen::Matrix2d* jacobian_wrt_point,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* jacobian_wrt_dist) const -> void {
  CHECK(point_out != nullptr);
  CHECK(dist_coeffs.size() == kNumParameters);
  const double x = point_in[0];
  const double y = point_in[1];
  const double& k1 = dist_coeffs[0];
  const double& k2 = dist_coeffs[1];
  const double& k3 = dist_coeffs[2];
  const double& k4 = dist_coeffs[3];
  const double rad = point_in.norm();

  const double theta = std::atan(rad);
  const double theta2 = theta * theta;
  const double theta4 = theta2 * theta2;
  const double theta6 = theta4 * theta2;
  const double theta8 = theta4 * theta4;
  const double thetad =
      theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
  if (jacobian_wrt_point != nullptr) {
    const double theta3 = theta2 * theta;
    const double theta5 = theta4 * theta;
    const double theta7 = theta6 * theta;
    const double mx2 = x * x;
    const double my2 = y * y;

    // MATLAB generated Jacobian
    const double duf_du =
        theta * 1.0 / rad *
            (k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8 + 1.0) +
        x * theta * 1.0 / rad *
            ((k2 * x * theta3 * 1.0 / rad * 4.0) / (mx2 + my2 + 1.0) +
             (k3 * x * theta5 * 1.0 / rad * 6.0) / (mx2 + my2 + 1.0) +
             (k4 * x * theta7 * 1.0 / rad * 8.0) / (mx2 + my2 + 1.0) +
             (k1 * x * theta * 1.0 / rad * 2.0) / (mx2 + my2 + 1.0)) +
        ((mx2) *
         (k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8 + 1.0)) /
            ((mx2 + my2) * (mx2 + my2 + 1.0)) -
        (mx2)*theta * 1.0 / std::pow(mx2 + my2, 3.0 / 2.0) *
            (k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8 + 1.0);

    const double duf_dv =
        x * theta * 1.0 / rad *
            ((k2 * y * theta3 * 1.0 / rad * 4.0) / (mx2 + my2 + 1.0) +
             (k3 * y * theta5 * 1.0 / rad * 6.0) / (mx2 + my2 + 1.0) +
             (k4 * y * theta7 * 1.0 / rad * 8.0) / (mx2 + my2 + 1.0) +
             (k1 * y * theta * 1.0 / rad * 2.0) / (mx2 + my2 + 1.0)) +
        (x * y *
         (k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8 + 1.0)) /
            ((mx2 + my2) * (mx2 + my2 + 1.0)) -
        x * y * theta * 1.0 / std::pow(mx2 + my2, 3.0 / 2.0) *
            (k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8 + 1.0);

    const double dvf_du =
        y * theta * 1.0 / rad *
            ((k2 * x * theta3 * 1.0 / rad * 4.0) / (mx2 + my2 + 1.0) +
             (k3 * x * theta5 * 1.0 / rad * 6.0) / (mx2 + my2 + 1.0) +
             (k4 * x * theta7 * 1.0 / rad * 8.0) / (mx2 + my2 + 1.0) +
             (k1 * x * theta * 1.0 / rad * 2.0) / (mx2 + my2 + 1.0)) +
        (x * y *
         (k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8 + 1.0)) /
            ((mx2 + my2) * (mx2 + my2 + 1.0)) -
        x * y * theta * 1.0 / std::pow(mx2 + my2, 3.0 / 2.0) *
            (k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8 + 1.0);

    const double dvf_dv =
        theta * 1.0 / rad *
            (k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8 + 1.0) +
        y * theta * 1.0 / rad *
            ((k2 * y * theta3 * 1.0 / rad * 4.0) / (mx2 + my2 + 1.0) +
             (k3 * y * theta5 * 1.0 / rad * 6.0) / (mx2 + my2 + 1.0) +
             (k4 * y * theta7 * 1.0 / rad * 8.0) / (mx2 + my2 + 1.0) +
             (k1 * y * theta * 1.0 / rad * 2.0) / (mx2 + my2 + 1.0)) +
        ((my2) *
         (k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8 + 1.0)) /
            ((mx2 + my2) * (mx2 + my2 + 1.0)) -
        (my2)*theta * 1.0 / std::pow(mx2 + my2, 3.0 / 2.0) *
            (k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8 + 1.0);
    *jacobian_wrt_point << duf_du, duf_dv, dvf_du, dvf_dv;
  }
  if (jacobian_wrt_dist != nullptr) {
    jacobian_wrt_dist->resize(2, kNumParameters);
    const double theta3 = theta2 * theta;
    const double theta5 = theta4 * theta;
    const double theta7 = theta6 * theta;
    const double theta9 = theta8 * theta;

    const double duf_dk1 = x * theta3 / rad;
    const double duf_dk2 = x * theta5 / rad;
    const double duf_dk3 = x * theta7 / rad;
    const double duf_dk4 = x * theta9 / rad;

    const double dvf_dk1 = y * theta3 / rad;
    const double dvf_dk2 = y * theta5 / rad;
    const double dvf_dk3 = y * theta7 / rad;
    const double dvf_dk4 = y * theta9 / rad;
    (*jacobian_wrt_dist) << duf_dk1, duf_dk2, duf_dk3, duf_dk4, dvf_dk1,
        dvf_dk2, dvf_dk3, dvf_dk4;
  }

  const double scaling = (rad > 1e-8) ? thetad / rad : 1.0;
  (*point_out)[0] = x * scaling;
  (*point_out)[1] = y * scaling;
}

}  // namespace dba
