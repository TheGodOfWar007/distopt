#pragma once

#include <ceres/ceres.h>

#include "distributed_bundle_adjustment/common.hpp"
#include "distributed_bundle_adjustment/data.hpp"

/// @brief add to dba namepsace
namespace dba {

/// @brief Optimization class declaration
class Optimization {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// @brief define the consensus algorithm to use
  enum ConsensusType { kNoConsensus = 0, kCentral = 1, kDecentral };
  using FrameDual = PoseDual<kNumIntrinsicParams, kNumDistortionParams>;

 public:
  Optimization() = delete;
  /// @brief constructor that takes in two varialbes
  /// @param data_ptr a shared Data ptr
  /// @param consensus_type the desired consensus algorithm to use
  Optimization(DataSharedPtr data_ptr, const ConsensusType& consensus_type =
                                           ConsensusType::kNoConsensus);
  ~Optimization();

  /// @brief setup Ceres problem for a given dataset
  /// @return 
  auto setupProblem() -> bool;

  auto performOptimization() -> bool;

  /// @brief update average estimates for Frame and MapPoint Duals
  /// @param frame_avgs 
  /// @param map_point_avgs 
  /// @param synchronized 
  /// @return 
  auto updateAverages(
      const std::unordered_map<uint64_t, FrameDual>& frame_avgs,
      const std::unordered_map<uint64_t, MapPointDual>& map_point_avgs,
      const bool synchronized = true) -> bool;

  /// @brief update Dual variables of optimization problem using decentralized algorithm
  /// @return 
  auto updateDuals() -> bool;

  /// @brief compute residuals for all constraints
  /// @return vector of residual values for the problem
  auto computeErrors() -> std::vector<double>;

 private:
  auto checkFrame(const uint64_t frame_id) -> double;

  // Store the consensus sigmas
  double sigma_map_points_;
  double sigma_intrinsics_;
  double sigma_distortion_;
  double sigma_rotation_;
  double sigma_translation_;

  std::unique_ptr<ceres::Problem> ceres_problem_;
  DataSharedPtr data_ptr_;
  const ConsensusType consensus_type_;
  std::unordered_set<ceres::ResidualBlockId> consensus_resids_;
};

using OptimizationSharedPtr = std::shared_ptr<Optimization>;
using OptimizationUniquePtr = std::unique_ptr<Optimization>;

}  // namespace dba
