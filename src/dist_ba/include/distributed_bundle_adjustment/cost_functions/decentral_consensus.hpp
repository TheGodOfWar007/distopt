#pragma once

#include <ceres/sized_cost_function.h>
#include <Eigen/Dense>
#include <memory>

#include "distributed_bundle_adjustment/common.hpp"

namespace dba {

namespace cost_functions {

template <int N>
class DecentralEuclideanConsensus : public ceres::SizedCostFunction<N, N> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:
  DecentralEuclideanConsensus() = delete;
  DecentralEuclideanConsensus(const double lambda,
                              const VectorOfVectorN<N>& duals)
      : lambda_(lambda),
        lambda_sqrt_(std::sqrt(lambda)),
        cardinality_(duals.size()),
        sqrt_cardinality_(std::sqrt(static_cast<double>(duals.size()))) {
    CHECK(!duals.empty());
    // Compute the average
    const double inv_card = 1.0 / static_cast<double>(cardinality_);
    dual_var_.setZero();
    for (const auto& d : duals) dual_var_ += d * inv_card;
    z_corr_ = dual_var_ / lambda_;
  };
  ~DecentralEuclideanConsensus(){};

  virtual auto Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const -> bool;

 private:
  const double lambda_;
  const double lambda_sqrt_;
  const int cardinality_;
  const double sqrt_cardinality_;
  Eigen::Matrix<double, N, 1> dual_var_;
  Eigen::Matrix<double, N, 1> z_corr_;
};

class DecentralRotationConsensus : public ceres::SizedCostFunction<3, 4> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:
  DecentralRotationConsensus() = delete;
  DecentralRotationConsensus(const double sigma,
                             const Eigen::Quaterniond& q_ref,
                             const VectorOfVector3& duals);

  virtual auto Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const -> bool;

 private:
  const double lambda_;
  const double lambda_sqrt_;
  const Eigen::Quaterniond q_ref_;
  const int cardinality_;
  const double sqrt_cardinality_;
  Eigen::Vector3d dual_var_;
  Eigen::Vector3d z_corr_;
};

}  // namespace cost_functions

}  // namespace dba

#include "decentral_consensus_impl.hpp"
