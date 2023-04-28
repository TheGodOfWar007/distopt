#include <glog/logging.h>

#include "distopt/cost_functions/central_consensus.hpp"
#include "distopt/utils.hpp"

namespace dba {

namespace cost_functions {

CentralRotationConsensus::CentralRotationConsensus(
    const double sigma, const Eigen::Quaterniond& q_ref,
    const Eigen::Vector3d& consensus, const Eigen::Vector3d& dual_var)
    : weight_(std::sqrt(sigma)),
      q_ref_(q_ref),
      consensus_(consensus),
      dual_var_(dual_var) {}

auto CentralRotationConsensus::Evaluate(double const* const* parameters,
                                        double* residuals,
                                        double** jacobians) const -> bool {
  const Eigen::Map<const Eigen::Quaterniond> q(parameters[0]);
  Eigen::Vector3d value;
  utils::rotmath::Minus(q, q_ref_, &value);
  Eigen::Map<Eigen::Vector3d> resid(residuals);
  resid = weight_ * (value - consensus_ + dual_var_);
  if (jacobians && jacobians[0]) {
    utils::QuaternionLocalParameterization quat_parameterization;
    Eigen::Matrix<double, 4, 3, Eigen::RowMajor> J_quat_local_param;
    quat_parameterization.ComputeJacobian(q.coeffs().data(),
                                          J_quat_local_param.data());
    Eigen::Matrix3d J_min = utils::rotmath::Gamma(value).inverse();

    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J(jacobians[0]);
    J = J_min * 4.0 * J_quat_local_param.transpose() * weight_;
  }
  return true;
}

}  // namespace cost_functions

}  // namespace dba
