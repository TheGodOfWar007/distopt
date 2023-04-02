#pragma once

#include <ceres/ceres.h>
#include <glog/logging.h>
#include <Eigen/Core>

namespace dba {

namespace utils {

namespace rotmath {

inline auto skew(const Eigen::Vector3d& vector) -> Eigen::Matrix3d {
  Eigen::Matrix3d matrix;
  matrix << 0.0, -vector[2], vector[1], vector[2], 0.0, -vector[0], -vector[1],
      vector[0], 0.0;
  return matrix;
}

inline auto Gamma(const Eigen::Vector3d& phi) -> Eigen::Matrix3d {
  const double phi_squared_norm = phi.squaredNorm();
  if (phi_squared_norm < 1e-6) {
    Eigen::Matrix3d gamma;
    gamma.setIdentity();
    gamma -= 0.5 * skew(phi);
    return gamma;
  }
  const double phi_norm = sqrt(phi_squared_norm);
  const Eigen::Matrix3d phi_skew(skew(phi));

  Eigen::Matrix3d gamma;
  gamma.setIdentity();
  gamma -= ((1.0 - std::cos(phi_norm)) / phi_squared_norm) * phi_skew;
  const double phi_cubed = (phi_norm * phi_squared_norm);
  gamma += ((phi_norm - std::sin(phi_norm)) / phi_cubed) * phi_skew * phi_skew;
  return gamma;
}

inline auto ExpMap(const Eigen::Vector3d& theta) -> Eigen::Quaterniond {
  const double theta_squared_norm = theta.squaredNorm();

  if (theta_squared_norm < 1e-6) {
    Eigen::Quaterniond q(1, theta(0) * 0.5, theta(1) * 0.5, theta(2) * 0.5);
    q.normalize();
    return q;
  }

  const double theta_norm = std::sqrt(theta_squared_norm);
  const Eigen::Vector3d q_imag =
      std::sin(theta_norm * 0.5) * theta / theta_norm;
  Eigen::Quaterniond q(std::cos(theta_norm * 0.5), q_imag(0), q_imag(1),
                       q_imag(2));
  return q;
}

inline auto LogMap(const Eigen::Quaterniond& q) -> Eigen::Vector3d {
  const Eigen::Block<const Eigen::Vector4d, 3, 1> q_imag = q.vec();
  const double q_imag_squared_norm = q_imag.squaredNorm();

  if (q_imag_squared_norm < 1e-6) {
    return 2 * std::copysign(1, q.w()) * q_imag;
  }

  const double q_imag_norm = std::sqrt(q_imag_squared_norm);
  Eigen::Vector3d q_log =
      2 * std::atan2(q_imag_norm, q.w()) * q_imag / q_imag_norm;
  return q_log;
}

// Implements the boxplus operator
// q_res = q boxplus delta
inline auto Plus(const Eigen::Ref<const Eigen::Vector4d>& q,
                 const Eigen::Ref<const Eigen::Vector3d>& delta,
                 Eigen::Quaterniond* q_res) -> void {
  CHECK_NOTNULL(q_res);
  const Eigen::Map<const Eigen::Quaterniond> p_mapped(q.data());
  *q_res = p_mapped * ExpMap(delta);
}

inline auto Plus(const Eigen::Quaterniond& q,
                 const Eigen::Ref<const Eigen::Vector3d>& delta,
                 Eigen::Quaterniond* q_res) -> void {
  CHECK_NOTNULL(q_res);
  *q_res = q * ExpMap(delta);
}

// Implements the boxminus operator
// p_minus_q = p boxminus q
inline auto Minus(const Eigen::Quaterniond& p, const Eigen::Quaterniond& q,
                  Eigen::Vector3d* p_minus_q) -> void {
  CHECK_NOTNULL(p_minus_q);
  Eigen::Quaterniond delta_q = q.inverse() * p;
  if (delta_q.w() < 0) delta_q.coeffs() = -delta_q.coeffs();
  *p_minus_q = LogMap(delta_q);
}

// Implements small angle approximation
// q = [1.0, 0.5*theta]
inline auto DeltaQ(const Eigen::Vector3d& theta) -> Eigen::Quaterniond {
  Eigen::Vector3d half_theta = theta / 2.0;
  Eigen::Quaterniond dq(1.0, half_theta[0], half_theta[1], half_theta[2]);
  return dq;
}

}  // namespace rotmath

class QuaternionLocalParameterization : public ceres::LocalParameterization {
 public:
  virtual auto Plus(const double* x, const double* delta,
                    double* x_plus_delta) const -> bool {
    Eigen::Map<Eigen::Quaterniond> q_res(x_plus_delta);
    Eigen::Map<const Eigen::Vector4d> q_curr(x);
    Eigen::Map<const Eigen::Vector3d> delta_curr(delta);

    Eigen::Quaterniond q_tmp;
    rotmath::Plus(q_curr, delta_curr, &q_tmp);

    q_res = q_tmp;
    return true;
  }

  virtual auto ComputeJacobian(const double* x, double* jacobian) const
      -> bool {
    Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> j(jacobian);
    const Eigen::Map<const Eigen::Quaterniond> quat_x(x);
    // Row 0
    jacobian[0] = quat_x.w() * 0.5;
    jacobian[1] = -quat_x.z() * 0.5;
    jacobian[2] = quat_x.y() * 0.5;
    // Row1
    jacobian[3] = quat_x.z() * 0.5;
    jacobian[4] = quat_x.w() * 0.5;
    jacobian[5] = -quat_x.x() * 0.5;
    // Row2
    jacobian[6] = -quat_x.y() * 0.5;
    jacobian[7] = quat_x.x() * 0.5;
    jacobian[8] = quat_x.w() * 0.5;
    // Row3
    jacobian[9] = -quat_x.x() * 0.5;
    jacobian[10] = -quat_x.y() * 0.5;
    jacobian[11] = -quat_x.z() * 0.5;
    return true;
  }

  virtual auto GlobalSize() const -> int { return 4; }
  virtual auto LocalSize() const -> int { return 3; }
};

}  // namespace utils

}  // namespace dba
