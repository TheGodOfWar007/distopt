namespace dba {

namespace cost_functions {

template <int N>
auto DecentralEuclideanConsensus<N>::Evaluate(double const* const* parameters,
                                              double* residuals,
                                              double** jacobians) const
    -> bool {
  Eigen::Map<const Eigen::Matrix<double, N, 1>> value(parameters[0]);
  Eigen::Map<Eigen::Matrix<double, N, 1>> resid_map(residuals);
  resid_map = (z_corr_ + value) * lambda_sqrt_ * sqrt_cardinality_;
  if (jacobians && jacobians[0]) {
    if (N > 1) {
      Eigen::Map<Eigen::Matrix<double, N, N, Eigen::RowMajor>> J(jacobians[0]);
      J = lambda_sqrt_ * sqrt_cardinality_ *
          Eigen::Matrix<double, N, N>::Identity();
    } else {
      jacobians[0][0] = lambda_sqrt_ * sqrt_cardinality_;
    }
  }
  return true;
}

}  // namespace cost_functions

}  // namespace dba
