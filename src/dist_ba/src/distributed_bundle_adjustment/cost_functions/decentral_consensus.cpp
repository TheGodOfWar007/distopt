#include <glog/logging.h>

#include "distributed_bundle_adjustment/cost_functions/decentral_consensus.hpp"
#include "distributed_bundle_adjustment/utils.hpp"

namespace dba {

namespace cost_functions {

DecentralRotationConsensus::DecentralRotationConsensus(
    const double lambda, const Eigen::Quaterniond& q_ref,
    const VectorOfVector3& duals)
    : lambda_(lambda),
      lambda_sqrt_(std::sqrt(lambda)),
      q_ref_(q_ref),
      cardinality_(duals.size()),
      sqrt_cardinality_(std::sqrt(static_cast<double>(duals.size()))) {
  CHECK(!duals.empty());
  // Compute the average
  const double inv_card = 1.0 / static_cast<double>(cardinality_);
  dual_var_.setZero();
  for (const auto& d : duals) dual_var_ += d * inv_card;
  z_corr_ = dual_var_ / lambda_;
}

auto DecentralRotationConsensus::Evaluate(double const* const* parameters,
                                          double* residuals,
                                          double** jacobians) const -> bool {
  const Eigen::Map<const Eigen::Quaterniond> q(parameters[0]);
  Eigen::Vector3d value;
  utils::rotmath::Minus(q, q_ref_, &value);
  Eigen::Map<Eigen::Vector3d> resid(residuals);
  resid = (z_corr_ + value) * lambda_sqrt_ * sqrt_cardinality_;
  if (jacobians && jacobians[0]) {
    utils::QuaternionLocalParameterization quat_parameterization;
    Eigen::Matrix<double, 4, 3, Eigen::RowMajor> J_quat_local_param;
    quat_parameterization.ComputeJacobian(q.coeffs().data(),
                                          J_quat_local_param.data());
    Eigen::Matrix3d J_min = utils::rotmath::Gamma(value).inverse();

    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J(jacobians[0]);
    J = J_min * 4.0 * J_quat_local_param.transpose() * lambda_sqrt_ *
        sqrt_cardinality_;
  }
  return true;
}

}  // namespace cost_functions

}  // namespace dba
