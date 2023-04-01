namespace dba {

namespace cost_functions {

template <int N>
auto CentralEuclideanConsensus<N>::Evaluate(double const* const* parameters,
                                            double* residuals,
                                            double** jacobians) const -> bool {
  Eigen::Map<const Eigen::Matrix<double, N, 1>> value(parameters[0]);
  Eigen::Map<Eigen::Matrix<double, N, 1>> resid_map(residuals);
  resid_map = (value - consensus_ + dual_var_) * weight_;
  if (jacobians && jacobians[0]) {
    if (N > 1) {
      Eigen::Map<Eigen::Matrix<double, N, N, Eigen::RowMajor>> J(jacobians[0]);
      J = weight_ * Eigen::Matrix<double, N, N>::Identity();
    } else {
      jacobians[0][0] = weight_;
    }
  }
  return true;
}

}  // namespace cost_functions

}  // namespace dba
