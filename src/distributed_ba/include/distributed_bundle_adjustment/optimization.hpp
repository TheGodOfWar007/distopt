#pragma once

#include <ceres/ceres.h>

#include "distributed_bundle_adjustment/common.hpp"
#include "distributed_bundle_adjustment/data.hpp"

namespace dba {

class Optimization {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum ConsensusType { kNoConsensus = 0, kCentral = 1, kDecentral };
  using FrameDual = PoseDual<kNumIntrinsicParams, kNumDistortionParams>;

 public:
  Optimization() = delete;
  Optimization(DataSharedPtr data_ptr, const ConsensusType& consensus_type =
                                           ConsensusType::kNoConsensus);
  ~Optimization();

  auto setupProblem() -> bool;

  auto performOptimization() -> bool;

  auto updateAverages(
      const std::unordered_map<uint64_t, FrameDual>& frame_avgs,
      const std::unordered_map<uint64_t, MapPointDual>& map_point_avgs,
      const bool synchronized = true) -> bool;

  auto updateDuals() -> bool;

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
