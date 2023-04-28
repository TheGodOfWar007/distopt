#pragma once

#include <ceres/sized_cost_function.h>
#include <Eigen/Dense>
#include <memory>

namespace dba {

namespace cost_functions {

template <int N>
class CentralEuclideanConsensus : public ceres::SizedCostFunction<N, N> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:
  CentralEuclideanConsensus() = delete;
  CentralEuclideanConsensus(const double sigma,
                            const Eigen::Matrix<double, N, 1>& consensus,
                            const Eigen::Matrix<double, N, 1>& dual_var)
      : weight_(std::sqrt(sigma)), consensus_(consensus), dual_var_(dual_var){};
  ~CentralEuclideanConsensus(){};

  virtual auto Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const -> bool;

 private:
  double weight_;
  Eigen::Matrix<double, N, 1> consensus_;
  Eigen::Matrix<double, N, 1> dual_var_;
};

class CentralRotationConsensus : public ceres::SizedCostFunction<3, 4> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:
  CentralRotationConsensus() = delete;
  CentralRotationConsensus(const double sigma, const Eigen::Quaterniond& q_ref,
                           const Eigen::Vector3d& consensus,
                           const Eigen::Vector3d& dual_var);

  virtual auto Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const -> bool;

 private:
  double weight_;
  const Eigen::Quaterniond q_ref_;
  Eigen::Vector3d consensus_;
  Eigen::Vector3d dual_var_;
};

}  // namespace cost_functions

}  // namespace dba

#include "central_consensus_impl.hpp"
