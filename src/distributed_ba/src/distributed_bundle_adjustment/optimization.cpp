#include <ceres/gradient_checker.h>
#include <ceres/loss_function.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <fstream>

#include "distributed_bundle_adjustment/cost_functions/central_consensus.hpp"
#include "distributed_bundle_adjustment/cost_functions/decentral_consensus.hpp"
#include "distributed_bundle_adjustment/cost_functions/reprojection_error.hpp"
#include "distributed_bundle_adjustment/equidistant_distortion.hpp"
#include "distributed_bundle_adjustment/optimization.hpp"
#include "distributed_bundle_adjustment/pinhole_camera.hpp"
#include "distributed_bundle_adjustment/rad_distortion.hpp"
#include "distributed_bundle_adjustment/radtan_distortion.hpp"
#include "distributed_bundle_adjustment/simple_pinhole_camera.hpp"

DEFINE_double(alpha, 0.0, "The over-relaxation parameter");
DEFINE_int32(num_ceres_threads, 4, "The number of solver threads for ceres");
DEFINE_int32(num_ceres_iter, 10, "The number of iterations for ceres");
DEFINE_bool(self_adaptation, false,
            "Whether the consensus penalty should be adapted automatically "
            "(only for sync version so far)");
DEFINE_double(observation_sigma, 1.0,
              "The expected standard deviation of the visual observations");

DECLARE_double(alpha_map_points);
DECLARE_double(alpha_intrinsics);
DECLARE_double(alpha_distortion);
DECLARE_double(alpha_rotation);
DECLARE_double(alpha_translation);

namespace dba {

/// @brief constructor for Optimization class
/// @param data_ptr shared Data Ptr to data to use
/// @param consensus_type consensus algorithm to use
Optimization::Optimization(DataSharedPtr data_ptr,
                           const ConsensusType& consensus_type)
    : ceres_problem_(nullptr),
      data_ptr_(data_ptr),
      consensus_type_(consensus_type) {
  // Compute the sigmas
  const size_t num_map_points = data_ptr_->getGlobalNumberOfMapPoints();
  const size_t num_frames = data_ptr_->getGlobalNumberOfFrames();
  const size_t num_observations = data_ptr_->getGlobalNumberOfObservations();
  sigma_map_points_ = FLAGS_alpha_map_points * num_observations /
                      static_cast<double>(num_map_points);
  sigma_intrinsics_ = FLAGS_alpha_intrinsics * num_observations /
                      static_cast<double>(num_frames);
  sigma_distortion_ = FLAGS_alpha_distortion * num_observations /
                      static_cast<double>(num_frames);
  sigma_rotation_ =
      FLAGS_alpha_rotation * num_observations / static_cast<double>(num_frames);
  sigma_translation_ = FLAGS_alpha_translation * num_observations /
                       static_cast<double>(num_frames);
}

Optimization::~Optimization() {}

/// @brief setup the optimization problem
/// @return 
auto Optimization::setupProblem() -> bool {
  const auto neighbors = data_ptr_->getNeighbors();
  if (ceres_problem_ != nullptr) return true;
  const auto frame_ids = data_ptr_->getFrameIds();
  if (frame_ids.empty()) return false;
  const auto map_point_ids = data_ptr_->getMapPointIds();
  if (map_point_ids.empty()) return false;
  ceres::Problem::Options prob_opts;
  prob_opts.enable_fast_removal = true;
  ceres_problem_ = std::make_unique<ceres::Problem>(prob_opts);
  consensus_resids_.clear();
  consensus_resids_.reserve(frame_ids.size() * 4 + map_point_ids.size());

  // Add the Map points to the problem
  for (const uint64_t id : map_point_ids) {
    auto map_point_ptr = data_ptr_->getMapPoint(id);
    if (map_point_ptr == nullptr) return false;
    ceres_problem_->AddParameterBlock(map_point_ptr->position_.data(), 3);
    if (consensus_type_ == ConsensusType::kCentral) {
      ceres::CostFunction* map_point_consensus =
          new cost_functions::CentralEuclideanConsensus<3>(
              sigma_map_points_, map_point_ptr->position_,
              Eigen::Vector3d::Zero());
      const auto resid_id = ceres_problem_->AddResidualBlock(
          map_point_consensus, NULL, map_point_ptr->position_.data());
      consensus_resids_.insert(resid_id);
    } else if (consensus_type_ == ConsensusType::kDecentral) {
      // TODO: Evaluate whether we should introduce a different error here or
      // not. At the moment it does not seem reasonable.
      const double lambda_map_point = map_point_ptr->lambda_;
      std::vector<MapPointDual> duals_map_point_async;
      bool has_neigh = false;
      for (auto n_id : neighbors) {
        if (n_id == data_ptr_->getGraphId()) continue;
        if (map_point_ptr->isNeighborInvalid(n_id)) continue;
        MapPointDual dual;
        if (map_point_ptr->getCommDual(n_id, dual)) {
          duals_map_point_async.push_back(dual);
          has_neigh = true;
        }
      }
      if (!has_neigh) {
        MapPointDual dual;
        for (size_t i = 0; i < dual.getSize(); ++i) {
          dual[i] = -map_point_ptr->position_[i] * lambda_map_point;
        }
        duals_map_point_async.push_back(dual);
      }
      VectorOfVector3 duals_map_point;
      for (const auto& d : duals_map_point_async) {
        Eigen::Vector3d dual(d[0], d[1], d[2]);
        duals_map_point.push_back(dual);
      }
      ceres::CostFunction* map_point_consensus =
          new cost_functions::DecentralEuclideanConsensus<3>(lambda_map_point,
                                                             duals_map_point);
      const auto resid_id = ceres_problem_->AddResidualBlock(
          map_point_consensus, NULL, map_point_ptr->position_.data());
      consensus_resids_.insert(resid_id);
    }
  }

  // Add the frames and their observations to the problem
  ceres::LossFunction* robust_loss = NULL;
  if (consensus_type_ != ConsensusType::kNoConsensus) {
    new ceres::HuberLoss(4.0);
  }
  ceres::LocalParameterization* local_quat_param =
      new utils::QuaternionLocalParameterization();
  int count = 0;
  int bad_count = 0;
  for (const uint64_t id : frame_ids) {
    auto frame_ptr = data_ptr_->getFrame(id);
    if (frame_ptr == nullptr) return false;
    if (!frame_ptr->is_valid_) continue;
    auto camera_ptr = frame_ptr->getCamera();
    const auto cam_type = frame_ptr->getCameraType();
    const auto dist_type = frame_ptr->getDistortionType();
    ceres_problem_->AddParameterBlock(frame_ptr->p_W_C_.data(), 3);
    ceres_problem_->AddParameterBlock(frame_ptr->q_W_C_.coeffs().data(), 4,
                                      local_quat_param);
    FrameDual duals_frame;
    std::vector<FrameDual> duals_frame_async;

    const double lambda_translation = frame_ptr->lambda_trans_;
    const double lambda_rotation = frame_ptr->lambda_rot_;
    const double lambda_intrinsics = frame_ptr->lambda_intr_;
    const double lambda_distortion = frame_ptr->lambda_dist_;
    if (consensus_type_ == ConsensusType::kCentral) {
      CHECK(frame_ptr->getCentralDual(duals_frame));
      ceres::CostFunction* translation_consensus =
          new cost_functions::CentralEuclideanConsensus<3>(
              sigma_translation_, frame_ptr->p_W_C_,
              Eigen::Matrix<double, 3, 1>::Zero());
      const auto trans_resid_id = ceres_problem_->AddResidualBlock(
          translation_consensus, NULL, frame_ptr->p_W_C_.data());
      consensus_resids_.insert(trans_resid_id);
      ceres::CostFunction* rotation_consensus =
          new cost_functions::CentralRotationConsensus(
              sigma_rotation_, frame_ptr->getReferenceRotation(),
              Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
      const auto rot_resid_id = ceres_problem_->AddResidualBlock(
          rotation_consensus, NULL, frame_ptr->q_W_C_.coeffs().data());
      consensus_resids_.insert(rot_resid_id);
    } else if (consensus_type_ == ConsensusType::kDecentral) {
      bool has_neigh = false;
      for (auto n_id : neighbors) {
        if (n_id == data_ptr_->getGraphId()) continue;
        FrameDual dual;
        if (frame_ptr->getCommDual(n_id, dual)) {
          duals_frame_async.push_back(dual);
          has_neigh = true;
        }
      }
      // In case we have no neighbor, we simply "fake" a dual as done in the
      // synchronous version
      if (!has_neigh) {
        FrameDual dual(id);
        Eigen::Map<Eigen::Vector3d> dual_pos(dual.getPosition());
        dual_pos = -lambda_translation * frame_ptr->p_W_C_;
        Eigen::Map<Eigen::Vector3d> dual_rot(dual.getRotation());
        dual_rot = Eigen::Vector3d::Zero();
        Eigen::Map<Eigen::Matrix<double, kNumIntrinsicParams, 1>> dual_intr(
            dual.getIntrisincs());
        dual_intr = -lambda_intrinsics * frame_ptr->intrinsics_;
        Eigen::Map<Eigen::Matrix<double, kNumDistortionParams, 1>> dual_dist(
            dual.getDistortion());
        dual_dist = -lambda_distortion * frame_ptr->dist_coeffs_;
        // TODO: Evaluate whether we really should add such a dual!
        duals_frame_async.push_back(dual);
      }
      if (!duals_frame_async.empty()) {
        // Transform the duals to the rotation and translation
        VectorOfVector3 duals_trans, duals_rot;
        for (auto& d : duals_frame_async) {
          Eigen::Vector3d dual_trans(d[0], d[1], d[2]);
          duals_trans.push_back(dual_trans);
          Eigen::Vector3d dual_rot(d[3], d[4], d[5]);
          duals_rot.push_back(dual_rot);
        }
        ceres::CostFunction* translation_consensus =
            new cost_functions::DecentralEuclideanConsensus<3>(
                lambda_translation, duals_trans);
        const auto trans_resid_id = ceres_problem_->AddResidualBlock(
            translation_consensus, NULL, frame_ptr->p_W_C_.data());
        consensus_resids_.insert(trans_resid_id);
        ceres::CostFunction* rotation_consensus =
            new cost_functions::DecentralRotationConsensus(
                lambda_rotation, frame_ptr->getReferenceRotation(), duals_rot);
        const auto rot_resid_id = ceres_problem_->AddResidualBlock(
            rotation_consensus, NULL, frame_ptr->q_W_C_.coeffs().data());
        consensus_resids_.insert(rot_resid_id);
      }
    }
    if (cam_type == Camera::Type::kPinholeSimple) {
      CHECK_EQ(kNumIntrinsicParams, SimplePinholeCamera::kNumParameters);
      ceres_problem_->AddParameterBlock(frame_ptr->intrinsics_.data(),
                                        SimplePinholeCamera::kNumParameters);
      if (consensus_type_ == ConsensusType::kCentral) {
        Eigen::Matrix<double, SimplePinholeCamera::kNumParameters, 1>
            intr_params;
        for (size_t i = 0; i < SimplePinholeCamera::kNumParameters; ++i)
          intr_params[i] = frame_ptr->intrinsics_[i];
        ceres::CostFunction* intrinsics_consensus =
            new cost_functions::CentralEuclideanConsensus<
                SimplePinholeCamera::kNumParameters>(
                sigma_intrinsics_, intr_params,
                Eigen::Matrix<double, SimplePinholeCamera::kNumParameters,
                              1>::Zero());
        const auto intr_res_id = ceres_problem_->AddResidualBlock(
            intrinsics_consensus, NULL, frame_ptr->intrinsics_.data());
        consensus_resids_.insert(intr_res_id);
      } else if (consensus_type_ == ConsensusType::kDecentral &&
                 !duals_frame_async.empty()) {
        VectorOfVectorN<SimplePinholeCamera::kNumParameters> duals_intr;
        for (auto& d : duals_frame_async) {
          Eigen::Matrix<double, SimplePinholeCamera::kNumParameters, 1>
              dual_intr;
          for (size_t i = 0; i < SimplePinholeCamera::kNumParameters; ++i)
            dual_intr[i] = d[6 + i];
          duals_intr.push_back(dual_intr);
        }
        ceres::CostFunction* intrinsics_consensus =
            new cost_functions::DecentralEuclideanConsensus<
                SimplePinholeCamera::kNumParameters>(lambda_intrinsics,
                                                     duals_intr);
        const auto intr_resid_id = ceres_problem_->AddResidualBlock(
            intrinsics_consensus, NULL, frame_ptr->intrinsics_.data());
        consensus_resids_.insert(intr_resid_id);
      }
    } else if (cam_type == Camera::Type::kPinhole) {
      CHECK_EQ(kNumIntrinsicParams, PinholeCamera::kNumParameters);
      CHECK_EQ(PinholeCamera::kNumParameters, frame_ptr->intrinsics_.size());
      ceres_problem_->AddParameterBlock(frame_ptr->intrinsics_.data(),
                                        PinholeCamera::kNumParameters);
      if (consensus_type_ == ConsensusType::kCentral) {
        Eigen::Matrix<double, PinholeCamera::kNumParameters, 1> intr_params;
        for (size_t i = 0; i < PinholeCamera::kNumParameters; ++i) {
          intr_params[i] = frame_ptr->intrinsics_[i];
        }
        ceres::CostFunction* intrinsics_consensus =
            new cost_functions::CentralEuclideanConsensus<4>(
                sigma_intrinsics_, intr_params,
                Eigen::Matrix<double, PinholeCamera::kNumParameters,
                              1>::Zero());
        const auto intr_resid_id = ceres_problem_->AddResidualBlock(
            intrinsics_consensus, NULL, frame_ptr->intrinsics_.data());
        consensus_resids_.insert(intr_resid_id);
      } else if (consensus_type_ == ConsensusType::kDecentral &&
                 !duals_frame_async.empty()) {
        VectorOfVectorN<PinholeCamera::kNumParameters> duals_intr;
        for (auto& d : duals_frame_async) {
          Eigen::Matrix<double, PinholeCamera::kNumParameters, 1> dual_intr;
          for (size_t i = 0; i < PinholeCamera::kNumParameters; ++i)
            dual_intr[i] = d[6 + i];
          duals_intr.push_back(dual_intr);
        }
        ceres::CostFunction* intrinsics_consensus =
            new cost_functions::DecentralEuclideanConsensus<
                PinholeCamera::kNumParameters>(lambda_intrinsics, duals_intr);
        const auto intr_resid_id = ceres_problem_->AddResidualBlock(
            intrinsics_consensus, NULL, frame_ptr->intrinsics_.data());
        consensus_resids_.insert(intr_resid_id);
      }
    }
    switch (dist_type) {
      case Distortion::Type::kRadDist: {
        CHECK_EQ(kNumDistortionParams, RadDistortion::kNumParameters);
        CHECK_EQ(RadDistortion::kNumParameters, frame_ptr->dist_coeffs_.size());
        ceres_problem_->AddParameterBlock(frame_ptr->dist_coeffs_.data(),
                                          RadDistortion::kNumParameters);
        if (consensus_type_ == ConsensusType::kCentral) {
          Eigen::Matrix<double, RadDistortion::kNumParameters, 1> dist_params;
          for (size_t i = 0; i < RadDistortion::kNumParameters; ++i)
            dist_params[i] = frame_ptr->dist_coeffs_[i];
          ceres::CostFunction* dist_consensus =
              new cost_functions::CentralEuclideanConsensus<
                  RadDistortion::kNumParameters>(
                  sigma_distortion_, dist_params,
                  Eigen::Matrix<double, RadDistortion::kNumParameters,
                                1>::Zero());
          const auto dist_resid_id = ceres_problem_->AddResidualBlock(
              dist_consensus, NULL, frame_ptr->dist_coeffs_.data());
          consensus_resids_.insert(dist_resid_id);
        } else if (consensus_type_ == ConsensusType::kDecentral &&
                   !duals_frame_async.empty()) {
          VectorOfVectorN<RadDistortion::kNumParameters> duals_dist;
          for (auto& d : duals_frame_async) {
            Eigen::Matrix<double, RadDistortion::kNumParameters, 1> dual_dist;
            for (size_t i = 0; i < RadDistortion::kNumParameters; ++i)
              dual_dist[i] = d[6 + kNumIntrinsicParams + i];
            duals_dist.push_back(dual_dist);
          }
          ceres::CostFunction* dist_consensus =
              new cost_functions::DecentralEuclideanConsensus<
                  RadDistortion::kNumParameters>(lambda_distortion, duals_dist);
          const auto dist_resid_id = ceres_problem_->AddResidualBlock(
              dist_consensus, NULL, frame_ptr->dist_coeffs_.data());
          consensus_resids_.insert(dist_resid_id);
        }
        break;
      }
      case Distortion::Type::kRadTanDist: {
        CHECK_EQ(kNumDistortionParams, RadTanDistortion::kNumParameters);
        CHECK_EQ(RadTanDistortion::kNumParameters,
                 frame_ptr->dist_coeffs_.size());
        ceres_problem_->AddParameterBlock(frame_ptr->dist_coeffs_.data(),
                                          RadTanDistortion::kNumParameters);
        if (consensus_type_ == ConsensusType::kCentral) {
          Eigen::Matrix<double, RadTanDistortion::kNumParameters, 1>
              dist_params;
          for (size_t i = 0; i < RadTanDistortion::kNumParameters; ++i)
            dist_params[i] = frame_ptr->dist_coeffs_[i];
          ceres::CostFunction* dist_consensus =
              new cost_functions::CentralEuclideanConsensus<
                  RadTanDistortion::kNumParameters>(
                  sigma_distortion_, dist_params,
                  Eigen::Matrix<double, RadTanDistortion::kNumParameters,
                                1>::Zero());
          const auto dist_resid_id = ceres_problem_->AddResidualBlock(
              dist_consensus, NULL, frame_ptr->dist_coeffs_.data());
          consensus_resids_.insert(dist_resid_id);
        } else if (consensus_type_ == ConsensusType::kDecentral &&
                   !duals_frame_async.empty()) {
          VectorOfVectorN<RadTanDistortion::kNumParameters> duals_dist;
          for (auto& d : duals_frame_async) {
            Eigen::Matrix<double, RadTanDistortion::kNumParameters, 1>
                dual_dist;
            for (size_t i = 0; i < RadTanDistortion::kNumParameters; ++i)
              dual_dist[i] = d[6 + kNumIntrinsicParams + i];
            duals_dist.push_back(dual_dist);
          }
          ceres::CostFunction* dist_consensus =
              new cost_functions::DecentralEuclideanConsensus<
                  RadTanDistortion::kNumParameters>(lambda_distortion,
                                                    duals_dist);
          const auto dist_resid_id = ceres_problem_->AddResidualBlock(
              dist_consensus, NULL, frame_ptr->dist_coeffs_.data());
          consensus_resids_.insert(dist_resid_id);
        }
        break;
      }
      case Distortion::Type::kEquiDist: {
        CHECK_EQ(kNumDistortionParams, EquidistantDistortion::kNumParameters);
        CHECK_EQ(EquidistantDistortion::kNumParameters,
                 frame_ptr->dist_coeffs_.size());
        ceres_problem_->AddParameterBlock(
            frame_ptr->dist_coeffs_.data(),
            EquidistantDistortion::kNumParameters);
        ceres_problem_->SetParameterBlockConstant(
            frame_ptr->dist_coeffs_.data());
        if (consensus_type_ == ConsensusType::kCentral) {
          Eigen::Matrix<double, EquidistantDistortion::kNumParameters, 1>
              dist_params;
          for (size_t i = 0; i < EquidistantDistortion::kNumParameters; ++i)
            dist_params[i] = frame_ptr->dist_coeffs_[i];
          ceres::CostFunction* dist_consensus =
              new cost_functions::CentralEuclideanConsensus<
                  EquidistantDistortion::kNumParameters>(
                  sigma_distortion_, dist_params,
                  Eigen::Matrix<double, EquidistantDistortion::kNumParameters,
                                1>::Zero());
          const auto dist_resid_id = ceres_problem_->AddResidualBlock(
              dist_consensus, NULL, frame_ptr->dist_coeffs_.data());
          consensus_resids_.insert(dist_resid_id);
        } else if (consensus_type_ == ConsensusType::kDecentral &&
                   !duals_frame_async.empty()) {
          VectorOfVectorN<EquidistantDistortion::kNumParameters> duals_dist;
          for (auto& d : duals_frame_async) {
            Eigen::Matrix<double, EquidistantDistortion::kNumParameters, 1>
                dual_dist;
            for (size_t i = 0; i < EquidistantDistortion::kNumParameters; ++i)
              dual_dist[i] = d[6 + kNumIntrinsicParams + i];
            duals_dist.push_back(dual_dist);
          }
          ceres::CostFunction* dist_consensus =
              new cost_functions::DecentralEuclideanConsensus<
                  EquidistantDistortion::kNumParameters>(lambda_distortion,
                                                         duals_dist);
          const auto dist_resid_id = ceres_problem_->AddResidualBlock(
              dist_consensus, NULL, frame_ptr->dist_coeffs_.data());
          consensus_resids_.insert(dist_resid_id);
        }
        break;
      }
    }

    const auto& observations = frame_ptr->getAllObservations();
    for (const auto obs_i : observations) {
      auto map_point_ptr = data_ptr_->getMapPoint(obs_i.mp_id);
      CHECK(map_point_ptr != nullptr);
      if (obs_i.frame_id != id) return false;
      ceres::CostFunction* reprojection_error;
      if (cam_type == Camera::Type::kPinholeSimple) {
        SimplePinholeCamera* derived_cam =
            static_cast<SimplePinholeCamera*>(camera_ptr.get());
        switch (dist_type) {
          case Distortion::Type::kRadDist: {
            reprojection_error =
                new cost_functions::ReprojectionError<SimplePinholeCamera,
                                                      RadDistortion>(
                    obs_i.obs, FLAGS_observation_sigma, derived_cam);
            break;
          }
          case Distortion::Type::kRadTanDist: {
            reprojection_error =
                new cost_functions::ReprojectionError<SimplePinholeCamera,
                                                      RadTanDistortion>(
                    obs_i.obs, FLAGS_observation_sigma, derived_cam);
            break;
          }
          case Distortion::Type::kEquiDist: {
            reprojection_error =
                new cost_functions::ReprojectionError<SimplePinholeCamera,
                                                      EquidistantDistortion>(
                    obs_i.obs, FLAGS_observation_sigma, derived_cam);
            break;
          }
        }
      } else if (cam_type == Camera::Type::kPinhole) {
        PinholeCamera* derived_cam =
            static_cast<PinholeCamera*>(camera_ptr.get());
        switch (dist_type) {
          case Distortion::Type::kRadDist: {
            reprojection_error =
                new cost_functions::ReprojectionError<PinholeCamera,
                                                      RadDistortion>(
                    obs_i.obs, FLAGS_observation_sigma, derived_cam);
            break;
          }
          case Distortion::Type::kRadTanDist: {
            reprojection_error =
                new cost_functions::ReprojectionError<PinholeCamera,
                                                      RadTanDistortion>(
                    obs_i.obs, FLAGS_observation_sigma, derived_cam);
            break;
          }
          case Distortion::Type::kEquiDist: {
            reprojection_error =
                new cost_functions::ReprojectionError<PinholeCamera,
                                                      EquidistantDistortion>(
                    obs_i.obs, FLAGS_observation_sigma, derived_cam);
            break;
          }
        }
      }
      //       The following code can be use for checking the gradients.
      //            ceres::NumericDiffOptions num_diff_opts;
      //       std::vector<const ceres::LocalParameterization*>
      //       local_parameterizations(
      //           {nullptr, local_quat_param, nullptr, nullptr, nullptr});
      //       ceres::GradientChecker grad_check(
      //           reprojection_error, &local_parameterizations, num_diff_opts);
      //       std::vector<double*> parameter_blocks(
      //           {map_point_ptr->position_.data(),
      //           frame_ptr->q_W_C_.coeffs().data(),
      //            frame_ptr->p_W_C_.data(), frame_ptr->intrinsics_.data(),
      //            frame_ptr->dist_coeffs_.data()});
      //       ceres::GradientChecker::ProbeResults results;
      //       if (!grad_check.Probe(parameter_blocks.data(), 1e-4, &results)) {
      //         ++bad_count;
      //       }

      const auto proj_resid_id = ceres_problem_->AddResidualBlock(
          reprojection_error, robust_loss, map_point_ptr->position_.data(),
          frame_ptr->q_W_C_.coeffs().data(), frame_ptr->p_W_C_.data(),
          frame_ptr->intrinsics_.data(), frame_ptr->dist_coeffs_.data());
      ++count;
    }
  }
  return true;
}

auto Optimization::performOptimization() -> bool {
  if (ceres_problem_ == nullptr) return false;
  ceres::Solver::Options solver_opts;
  solver_opts.num_threads = FLAGS_num_ceres_threads;
  solver_opts.max_num_iterations = FLAGS_num_ceres_iter;
  solver_opts.sparse_linear_algebra_library_type =
      ceres::SparseLinearAlgebraLibraryType::SUITE_SPARSE;
  solver_opts.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
  solver_opts.trust_region_strategy_type =
      ceres::TrustRegionStrategyType::DOGLEG;
  solver_opts.minimizer_progress_to_stdout = false;
  solver_opts.check_gradients = false;
  ceres::Solver::Summary summary;
  ceres::Solve(solver_opts, ceres_problem_.get(), &summary);
  const auto frame_ids = data_ptr_->getFrameIds();
  //  for (const auto id : frame_ids) {
  //    auto frame_ptr = data_ptr_->getFrame(id);
  //    CHECK(frame_ptr != nullptr);
  //    if (!frame_ptr->is_valid_) continue;
  //    const double err = checkFrame(id);
  //    if (err > 15.0) frame_ptr->is_valid_ = false;
  //  }
  std::cout << "summary (" << data_ptr_->getGraphId() << "):\n"
            << summary.BriefReport() << std::endl;
  return true;
}

/// @brief update Frame and MapPoint averages
/// @param frame_avgs 
/// @param map_point_avgs 
/// @param synchronized boolean value that presumably indicates if all Frames/MapPoints are synchronized
/// @return 
auto Optimization::updateAverages(
    const std::unordered_map<uint64_t, FrameDual>& frame_avgs,
    const std::unordered_map<uint64_t, MapPointDual>& map_point_avgs,
    const bool synchronized) -> bool {
  // Note that this function is specific for updating the error terms related to
  // the dual variables for a centralized approach.
  if (frame_avgs.empty() && map_point_avgs.empty()) {
    std::cout << "averages.empty()" << std::endl;
    return false;
  }
  if (ceres_problem_ == nullptr) {
    std::cout << "ceres_proble == nullptr" << std::endl;
    return false;
  }
  if (consensus_resids_.empty()) {
    std::cout << "consensus_resid.empty()" << std::endl;
    return false;
  }
  const auto frame_ids = data_ptr_->getFrameIds();
  if (frame_ids.empty()) {
    std::cout << "frame_ids.empty()" << std::endl;
    return false;
  }
  const auto map_point_ids = data_ptr_->getMapPointIds();
  if (map_point_ids.empty()) {
    std::cout << "map_point_ids.empty()" << std::endl;
    return false;
  }

  // Remove all consensus residual blocks and clear the consensus residual block ids
  for (const auto& id : consensus_resids_)
    ceres_problem_->RemoveResidualBlock(id);
  consensus_resids_.clear();

  // Re-add the map point terms 
  for (const uint64_t id : map_point_ids) {
    Eigen::Vector3d dual_var_vec;
    Eigen::Vector3d average_pos;
    auto map_point_ptr = data_ptr_->getMapPoint(id);
    CHECK(map_point_ptr != nullptr);
    // if synchronized and the average value is present
    if (map_point_avgs.count(id) && synchronized) {
      // Compute the primal residuals (local state - average_last)
      MapPointDual primal_res;
      for (size_t i = 0; i < 3; ++i) {
        primal_res[i] =
            map_point_ptr->position_[i] - map_point_ptr->average_state_[i];
      }
      const auto& avg = map_point_avgs.at(id);
      double s = 0.0, r = 0.0;
      for (size_t i = 0; i < primal_res.getSize(); ++i) {
        r += primal_res[i] * primal_res[i];
        const double tmp = (avg[i] - map_point_ptr->average_state_[i]) *
                           std::sqrt(sigma_map_points_);
        s += tmp * tmp;
      }
      r = std::sqrt(r);
      s = std::sqrt(s);
      const double mu1 = 10.0 / sigma_map_points_;
      const double mu2 = 10.0 * sigma_map_points_;
      double fact = 1.0;
      //      if (FLAGS_self_adaptation) {
      //        if (r > mu1 * s)
      //          fact = 2.0;
      //        else if (s > mu2 * r)
      //          fact = 0.5;
      //        else
      //          fact = 1.0;
      //      }
      map_point_ptr->sigma_ *= fact;
      MapPointDual dual(id);
      CHECK(map_point_ptr->getCentralDual(dual));
      for (size_t i = 0; i < dual.getSize(); ++i) {
        dual[i] += map_point_ptr->position_[i] - avg[i] * (1.0 + FLAGS_alpha);
        dual[i] /= fact;
        dual_var_vec[i] = dual[i];
        map_point_ptr->average_state_[i] = avg[i];
        average_pos[i] = avg[i];
      }
      map_point_ptr->setCentralDual(dual);
    } else if (map_point_avgs.count(id) && !synchronized) { // if the average is present but not synchronized
      // Retrieve Dual without updating average
      MapPointDual dual;
      CHECK(map_point_ptr->getCentralDual(dual));
      for (size_t i = 0; i < dual.getSize(); ++i) {
        dual_var_vec[i] = dual[i];
        average_pos[i] = map_point_ptr->average_state_[i];
      }
    } else { // if map point average is not present
      // Set average to current map point position and Dual variable to 0
      average_pos = map_point_ptr->position_;
      dual_var_vec.setZero();
    }

    // Create new consensus of map_point using CentralEuclideanConsensus cost function
    ceres::CostFunction* map_point_consensus =
        new cost_functions::CentralEuclideanConsensus<3>(
            map_point_ptr->sigma_, average_pos, dual_var_vec);
    // Add residual block
    const auto resid_id = ceres_problem_->AddResidualBlock(
        map_point_consensus, NULL, map_point_ptr->position_.data());
    // add residual ID to consensus_resids_
    consensus_resids_.insert(resid_id);
  }

  // Iterate over frame_ids:
  // If the frame is invalid, remove from Ceres problem
  // Else, give a camera ptr, camera frame, and distortion value

  for (const uint64_t id : frame_ids) {
    auto frame_ptr = data_ptr_->getFrame(id);
    if (frame_ptr == nullptr) {
      std::cout << "frame: " << id << " is not available" << std::endl;
      return false;
    }
    if (!frame_ptr->is_valid_) {
      if (ceres_problem_->HasParameterBlock(frame_ptr->p_W_C_.data()))
        ceres_problem_->RemoveParameterBlock(frame_ptr->p_W_C_.data());
      if (ceres_problem_->HasParameterBlock(frame_ptr->q_W_C_.coeffs().data()))
        ceres_problem_->RemoveParameterBlock(frame_ptr->q_W_C_.coeffs().data());
      if (ceres_problem_->HasParameterBlock(frame_ptr->intrinsics_.data()))
        ceres_problem_->RemoveParameterBlock(frame_ptr->intrinsics_.data());
      if (ceres_problem_->HasParameterBlock(frame_ptr->dist_coeffs_.data()))
        ceres_problem_->RemoveParameterBlock(frame_ptr->dist_coeffs_.data());
      continue;
    }
    auto camera_ptr = frame_ptr->getCamera();
    auto cam_type = frame_ptr->getCameraType();
    auto dist_type = frame_ptr->getDistortionType();

    Eigen::Vector3d avg_p_W_C, avg_q_W_C, dual_p_W_C, dual_q_W_C;
    Eigen::Matrix<double, kNumIntrinsicParams, 1> avg_intr, dual_intr;
    Eigen::Matrix<double, kNumDistortionParams, 1> avg_dist, dual_dist;
    if (frame_avgs.count(id) && synchronized) {
      // Compute the primal residuals (local state - average_last)
      FrameDual primal_res;
      Eigen::Vector3d state_q_W_C;
      utils::rotmath::Minus(frame_ptr->q_W_C_,
                            frame_ptr->getReferenceRotation(), &state_q_W_C);
      Eigen::Map<Eigen::Vector3d> avg_pos_map(
          frame_ptr->average_state_.getPosition());
      Eigen::Map<Eigen::Vector3d> primal_res_pos(primal_res.getPosition());
      primal_res_pos = frame_ptr->p_W_C_ - avg_pos_map;
      Eigen::Map<Eigen::Vector3d> avg_rot_map(
          frame_ptr->average_state_.getRotation());
      Eigen::Map<Eigen::Vector3d> primal_res_rot(primal_res.getRotation());
      primal_res_rot = state_q_W_C - avg_rot_map;
      Eigen::Map<Eigen::Matrix<double, kNumIntrinsicParams, 1>> avg_intr_map(
          frame_ptr->average_state_.getIntrisincs());
      Eigen::Map<Eigen::Matrix<double, kNumIntrinsicParams, 1>> primal_res_intr(
          primal_res.getIntrisincs());
      primal_res_intr = frame_ptr->intrinsics_ - avg_intr_map;
      Eigen::Map<Eigen::Matrix<double, kNumDistortionParams, 1>> avg_dist_map(
          frame_ptr->average_state_.getDistortion());
      Eigen::Map<Eigen::Matrix<double, kNumDistortionParams, 1>>
          primal_res_dist(primal_res.getDistortion());
      primal_res_dist = frame_ptr->dist_coeffs_ - avg_dist_map;
      const auto& avg = frame_avgs.at(id);

      FrameDual dual_res;
      // r : residual
      // t : pos, q : rot, i : intrinsics, d : distortion
      double r_t = 0.0, r_q = 0.0, r_i = 0.0, r_d = 0.0;
      // s : sigma term
      double s_t = 0.0, s_q = 0.0, s_i = 0.0, s_d = 0.0;
      for (size_t i = 0; i < dual_res.getSize(); ++i) {
        if (i < 3) {
          // Translation
          r_t += primal_res[i] * primal_res[i];
          const double tmp = (avg[i] - frame_ptr->average_state_[i]) *
                             std::sqrt(sigma_translation_);
          s_t += tmp * tmp;
        } else if (i >= 3 && i < 6) {
          // Rotation
          r_q += primal_res[i] * primal_res[i];
          const double tmp = (avg[i] - frame_ptr->average_state_[i]) *
                             std::sqrt(sigma_rotation_);
          s_q += tmp * tmp;
        } else if (i >= 6 && i < 6 + kNumIntrinsicParams) {
          // Intrinsics
          r_i += primal_res[i] * primal_res[i];
          const double tmp = (avg[i] - frame_ptr->average_state_[i]) *
                             std::sqrt(sigma_intrinsics_);
          s_i += tmp * tmp;
        } else {
          // Distortion
          r_d += primal_res[i] * primal_res[i];
          const double tmp = (avg[i] - frame_ptr->average_state_[i]) *
                             std::sqrt(sigma_distortion_);
          s_i += tmp * tmp;
        }
      }

      r_t = std::sqrt(r_t);
      s_t = std::sqrt(s_t);
      r_q = std::sqrt(r_q);
      s_q = std::sqrt(s_q);
      r_i = std::sqrt(r_i);
      s_i = std::sqrt(s_i);
      r_d = std::sqrt(r_d);
      s_d = std::sqrt(s_d);

      //      for (size_t i = 0; i < dual_res.getSize(); ++i) {
      //        primal_res[i] *= primal_res[i];
      //        dual_res[i] *= dual_res[i];
      //        frame_ptr->average_state_[i] = avg[i];
      //      }
      //      const double r_t =
      //          std::sqrt(primal_res[0] + primal_res[1] + primal_res[2]);
      //      const double r_q =
      //          std::sqrt(primal_res[3] + primal_res[4] + primal_res[5]);
      //      const double r_i = std::sqrt(primal_res[6]);
      //      const double r_d = std::sqrt(primal_res[7] + primal_res[8]);
      //      const double s_t = std::sqrt(dual_res[0] + dual_res[1] +
      //      dual_res[2]); const double s_q = std::sqrt(dual_res[3] +
      //      dual_res[4] + dual_res[5]); const double s_i =
      //      std::sqrt(dual_res[6]); const double s_d = std::sqrt(dual_res[7] +
      //      dual_res[8]);

      const double mu1_t = 10.0 / sigma_translation_;
      const double mu2_t = 10.0 * sigma_translation_;
      const double mu1_q = 10.0 / sigma_rotation_;
      const double mu2_q = 10.0 * sigma_rotation_;
      const double mu1_i = 10.0 / sigma_intrinsics_;
      const double mu2_i = 10.0 * sigma_intrinsics_;
      const double mu1_d = 10.0 / sigma_distortion_;
      const double mu2_d = 10.0 * sigma_distortion_;

      double fact_t = 1.0, fact_q = 1.0, fact_i = 1.0, fact_d = 1.0;
      if (FLAGS_self_adaptation) {
        if (r_t > mu1_t * s_t)
          fact_t = 2.0;
        else if (s_t > mu2_t * r_t)
          fact_t = 0.5;
        if (r_q > mu1_q * s_q)
          fact_q = 2.0;
        else if (s_q > mu2_q * r_q)
          fact_q = 0.5;
        if (r_i > mu1_i * s_i)
          fact_i = 2.0;
        else if (s_i > mu2_i * r_i)
          fact_i = 1.0;
        if (r_d > mu1_d * s_d)
          fact_d = 2.0;
        else if (s_d > mu2_d * r_d)
          fact_d = 0.5;
      }

      frame_ptr->sigma_trans_ *= fact_t;
      frame_ptr->sigma_rot_ *= fact_q;
      frame_ptr->sigma_intr_ *= fact_i;
      frame_ptr->sigma_dist_ *= fact_d;

      FrameDual dual;
      CHECK(frame_ptr->getCentralDual(dual));
      utils::rotmath::Minus(frame_ptr->q_W_C_,
                            frame_ptr->getReferenceRotation(), &state_q_W_C);

      // Update (?)
      // (9) (10)
      for (size_t i = 0; i < dual.getSize(); ++i) {
        frame_ptr->average_state_[i] = avg[i];
        if (i < 3) {
          // Translation resdual calculation
          dual[i] += (frame_ptr->p_W_C_[i] - avg[i]) * (1.0 + FLAGS_alpha);
          dual[i] /= fact_t;
          dual_p_W_C[i] = dual[i];
          avg_p_W_C[i] = avg[i];
        } else if (i >= 3 && i < 6) {
          // Rotation
          dual[i] += (state_q_W_C[i - 3] - avg[i]) * (1.0 + FLAGS_alpha);
          dual[i] /= fact_q;
          dual_q_W_C[i - 3] = dual[i];
          avg_q_W_C[i - 3] = avg[i];
        } else if (i >= 6 && i < 6 + kNumIntrinsicParams) {
          // Intrinsics
          dual[i] +=
              (frame_ptr->intrinsics_[i - 6] - avg[i]) * (1.0 + FLAGS_alpha);
          dual[i] /= fact_i;

          dual_intr[i - 6] = dual[i];
          avg_intr[i - 6] = avg[i];

        } else {
          // Distortion
          dual[i] +=
              (frame_ptr->dist_coeffs_[i - 6 - kNumIntrinsicParams] - avg[i]) *
              (1.0 + FLAGS_alpha);
          dual[i] /= fact_d;
          dual_dist[i - 6 - kNumIntrinsicParams] = dual[i];
          avg_dist[i - 6 - kNumIntrinsicParams] = avg[i];
        }
      }
      frame_ptr->setCentralDual(dual);
      //      std::cout << "diff_p: "
      //                << FLAGS_sigma_map_points *
      //                       (avg_p_W_C - frame_ptr->p_W_C_).norm()
      //                << std::endl;
      //      std::cout << "diff_intr: "
      //                << FLAGS_sigma_intrinsics *
      //                       (avg_intr - frame_ptr->intrinsics_).norm()
      //                << std::endl;
      //      std::cout << "diff_dist: "
      //                << FLAGS_sigma_distortion *
      //                       (avg_dist - frame_ptr->dist_coeffs_).norm()
      //                << std::endl;
    } else if (frame_avgs.count(id) && !synchronized) {
      FrameDual dual;
      CHECK(frame_ptr->getCentralDual(dual));
      for (size_t i = 0; i < dual.getSize(); ++i) {
        if (i < 3) {
          // Translation
          dual_p_W_C[i] = dual[i];
          avg_p_W_C[i] = frame_ptr->average_state_[i];
        } else if (i >= 3 && i < 6) {
          // Rotation
          dual_q_W_C[i - 3] = dual[i];
          avg_q_W_C[i - 3] = frame_ptr->average_state_[i];
        } else if (i >= 6 && i < 6 + kNumIntrinsicParams) {
          // Intrinsics
          dual_intr[i - 6] = dual[i];
          avg_intr[i - 6] = frame_ptr->average_state_[i];
        } else {
          // Distortion
          dual_dist[i - 6 - kNumIntrinsicParams] = dual[i];
          avg_dist[i - 6 - kNumIntrinsicParams] = frame_ptr->average_state_[i];
        }
      }
    } else {
      Eigen::Vector3d delta_q;
      utils::rotmath::Minus(frame_ptr->q_W_C_,
                            frame_ptr->getReferenceRotation(), &delta_q);
      dual_p_W_C.setZero();
      dual_q_W_C.setZero();
      dual_intr.setZero();
      dual_dist.setZero();
      avg_p_W_C = frame_ptr->p_W_C_;
      avg_q_W_C = delta_q;
      avg_intr = frame_ptr->intrinsics_;
      avg_dist = frame_ptr->dist_coeffs_;
    }
    // Translation consensus:
    ceres::CostFunction* translation_consensus =
        new cost_functions::CentralEuclideanConsensus<3>(
            frame_ptr->sigma_trans_, avg_p_W_C, dual_p_W_C);
    const auto trans_resid_id = ceres_problem_->AddResidualBlock(
        translation_consensus, NULL, frame_ptr->p_W_C_.data());

    consensus_resids_.insert(trans_resid_id);
    // Rotation consensus:
    ceres::CostFunction* rotation_consensus =
        new cost_functions::CentralRotationConsensus(
            frame_ptr->sigma_rot_, frame_ptr->getReferenceRotation(), avg_q_W_C,
            dual_q_W_C);
    const auto rot_resid_id = ceres_problem_->AddResidualBlock(
        rotation_consensus, NULL, frame_ptr->q_W_C_.coeffs().data());
    consensus_resids_.insert(rot_resid_id);
    if (cam_type == Camera::Type::kPinholeSimple) {
      Eigen::Matrix<double, SimplePinholeCamera::kNumParameters, 1> tmp_avg,
          tmp_dual;
      for (size_t i = 0; i < SimplePinholeCamera::kNumParameters; ++i) {
        tmp_avg[i] = avg_intr[i];
        tmp_dual[i] = dual_intr[i];
      }
      ceres::CostFunction* intrinsics_consensus =
          new cost_functions::CentralEuclideanConsensus<
              SimplePinholeCamera::kNumParameters>(frame_ptr->sigma_intr_,
                                                   tmp_avg, tmp_dual);
      const auto res_id = ceres_problem_->AddResidualBlock(
          intrinsics_consensus, NULL, frame_ptr->intrinsics_.data());
      consensus_resids_.insert(res_id);
    } else if (cam_type == Camera::Type::kPinhole) {
      Eigen::Matrix<double, PinholeCamera::kNumParameters, 1> tmp_avg, tmp_dual;
      for (size_t i = 0; i < PinholeCamera::kNumParameters; ++i) {
        tmp_avg[i] = avg_intr[i];
        tmp_dual[i] = dual_intr[i];
      }
      ceres::CostFunction* intrinsics_consensus =
          new cost_functions::CentralEuclideanConsensus<
              PinholeCamera::kNumParameters>(frame_ptr->sigma_intr_, tmp_avg,
                                             tmp_dual);
      const auto res_id = ceres_problem_->AddResidualBlock(
          intrinsics_consensus, NULL, frame_ptr->intrinsics_.data());
      consensus_resids_.insert(res_id);
    }
    switch (dist_type) {
      case Distortion::Type::kRadDist: {
        Eigen::Matrix<double, RadDistortion::kNumParameters, 1> tmp_avg,
            tmp_dual;
        for (size_t i = 0; i < RadDistortion::kNumParameters; ++i) {
          tmp_avg[i] = avg_dist[i];
          tmp_dual[i] = dual_dist[i];
        }
        ceres::CostFunction* dist_consensus =
            new cost_functions::CentralEuclideanConsensus<
                RadDistortion::kNumParameters>(frame_ptr->sigma_dist_, tmp_avg,
                                               tmp_dual);
        const auto resid_id = ceres_problem_->AddResidualBlock(
            dist_consensus, NULL, frame_ptr->dist_coeffs_.data());
        consensus_resids_.insert(resid_id);
        break;
      }
      case Distortion::Type::kRadTanDist: {
        Eigen::Matrix<double, RadTanDistortion::kNumParameters, 1> tmp_avg,
            tmp_dual;
        for (size_t i = 0; i < RadTanDistortion::kNumParameters; ++i) {
          tmp_avg[i] = avg_dist[i];
          tmp_dual[i] = dual_dist[i];
        }
        ceres::CostFunction* dist_consensus =
            new cost_functions::CentralEuclideanConsensus<
                RadTanDistortion::kNumParameters>(frame_ptr->sigma_dist_,
                                                  tmp_avg, tmp_dual);
        const auto resid_id = ceres_problem_->AddResidualBlock(
            dist_consensus, NULL, frame_ptr->dist_coeffs_.data());
        consensus_resids_.insert(resid_id);
        break;
      }
      case Distortion::Type::kEquiDist: {
        Eigen::Matrix<double, EquidistantDistortion::kNumParameters, 1> tmp_avg,
            tmp_dual;
        for (size_t i = 0; i < EquidistantDistortion::kNumParameters; ++i) {
          tmp_avg[i] = avg_dist[i];
          tmp_dual[i] = dual_dist[i];
        }
        ceres::CostFunction* dist_consensus =
            new cost_functions::CentralEuclideanConsensus<
                EquidistantDistortion::kNumParameters>(frame_ptr->sigma_dist_,
                                                       tmp_avg, tmp_dual);
        const auto resid_id = ceres_problem_->AddResidualBlock(
            dist_consensus, NULL, frame_ptr->dist_coeffs_.data());
        consensus_resids_.insert(resid_id);
        break;
      }
    }
  }

  return true;
}

/// @brief update Dual variables of optimization problem using decentralized algorithm
/// @return 
auto Optimization::updateDuals() -> bool {
  if (ceres_problem_ == nullptr) return false;
  if (consensus_resids_.empty()) return false;
  const auto frame_ids = data_ptr_->getFrameIds();
  if (frame_ids.empty()) return false;
  const auto map_point_ids = data_ptr_->getMapPointIds();
  if (map_point_ids.empty()) return false;
  for (const auto& id : consensus_resids_)
    ceres_problem_->RemoveResidualBlock(id);
  consensus_resids_.clear();
  const auto neighbors = data_ptr_->getNeighbors();

  // Update the Map points
  for (const uint64_t id : map_point_ids) {
    auto map_point_ptr = data_ptr_->getMapPoint(id);
    const double lambda_map_point = map_point_ptr->lambda_;
    if (map_point_ptr == nullptr) return false;
    std::vector<MapPointDual> duals_map_point_async;
    bool has_neigh = false;
    for (auto n_id : neighbors) {
      if (n_id == data_ptr_->getGraphId()) continue;
      if (map_point_ptr->isNeighborInvalid(n_id)) continue;
      MapPointDual dual;
      if (map_point_ptr->getCommDual(n_id, dual)) {
        duals_map_point_async.push_back(dual);
        has_neigh = true;
      }
    }
    if (!has_neigh) {
      MapPointDual dual;
      for (size_t i = 0; i < dual.getSize(); ++i) {
        dual[i] = -map_point_ptr->position_[i] * lambda_map_point;
      }
      duals_map_point_async.push_back(dual);
    }
    VectorOfVector3 duals_map_point;
    for (const auto& d : duals_map_point_async) {
      Eigen::Vector3d dual(d[0], d[1], d[2]);
      duals_map_point.push_back(dual);
    }
    ceres::CostFunction* map_point_consensus =
        new cost_functions::DecentralEuclideanConsensus<3>(lambda_map_point,
                                                           duals_map_point);
    const auto resid_id = ceres_problem_->AddResidualBlock(
        map_point_consensus, NULL, map_point_ptr->position_.data());
    consensus_resids_.insert(resid_id);
  }

  // Add the frames and their observations to the problem
  ceres::LocalParameterization* local_quat_param =
      new utils::QuaternionLocalParameterization();
  int count = 0;
  int bad_count = 0;
  for (const uint64_t id : frame_ids) {
    auto frame_ptr = data_ptr_->getFrame(id);
    if (frame_ptr == nullptr) return false;
    const double lambda_translation = frame_ptr->lambda_trans_;
    const double lambda_rotation = frame_ptr->lambda_rot_;
    const double lambda_intrinsics = frame_ptr->lambda_intr_;
    const double lambda_distortion = frame_ptr->lambda_dist_;
    if (!frame_ptr->is_valid_) {
      LOG(FATAL) << "This should not happen at the moment";
      if (ceres_problem_->HasParameterBlock(frame_ptr->p_W_C_.data()))
        ceres_problem_->RemoveParameterBlock(frame_ptr->p_W_C_.data());
      if (ceres_problem_->HasParameterBlock(frame_ptr->q_W_C_.coeffs().data()))
        ceres_problem_->RemoveParameterBlock(frame_ptr->q_W_C_.coeffs().data());
      if (ceres_problem_->HasParameterBlock(frame_ptr->intrinsics_.data()))
        ceres_problem_->RemoveParameterBlock(frame_ptr->intrinsics_.data());
      if (ceres_problem_->HasParameterBlock(frame_ptr->dist_coeffs_.data()))
        ceres_problem_->RemoveParameterBlock(frame_ptr->dist_coeffs_.data());
      continue;
    }
    auto camera_ptr = frame_ptr->getCamera();
    auto cam_type = frame_ptr->getCameraType();
    auto dist_type = frame_ptr->getDistortionType();
    std::vector<FrameDual> duals_frame_async;
    bool has_neigh = false;
    for (auto n_id : neighbors) {
      if (n_id == data_ptr_->getGraphId()) continue;
      if (frame_ptr->isNeighborInvalid(n_id)) {
        LOG(FATAL) << "This should not happen at the moment";
        continue;
      }
      FrameDual dual;
      if (frame_ptr->getCommDual(n_id, dual)) {
        duals_frame_async.push_back(dual);
        has_neigh = true;
      }
    }
    // In case we have no neighbor, we simply "fake" a dual as done in the
    // synchronous version
    if (!has_neigh) {
      // TODO: Evaluate whether we really should add such a dual!
      FrameDual dual;
      const Eigen::Quaterniond q_ref = frame_ptr->getReferenceRotation();
      Eigen::Vector3d delta_q;
      utils::rotmath::Minus(frame_ptr->q_W_C_, q_ref, &delta_q);
      for (size_t i = 0; i < dual.getSize(); ++i) {
        if (i < 3) {
          // Translation
          dual[i] = -frame_ptr->p_W_C_[i] * lambda_translation;
        } else if (i >= 3 && i < 6) {
          // Rotation
          dual[i] = -delta_q[i - 3] * lambda_rotation;
        } else if (i >= 6 && i < 6 + kNumIntrinsicParams) {
          // Intrinsics
          dual[i] = -frame_ptr->intrinsics_[i - 6] * lambda_intrinsics;
        } else {
          // Distortion
          dual[i] = -frame_ptr->dist_coeffs_[i - 6 - kNumIntrinsicParams] *
                    lambda_distortion;
        }
      }
      duals_frame_async.push_back(dual);
    }
    if (duals_frame_async.empty()) continue;
    // Transform the duals to the rotation and translation
    VectorOfVector3 duals_trans, duals_rot;
    for (const auto& d : duals_frame_async) {
      Eigen::Vector3d dual_trans(d[0], d[1], d[2]);
      duals_trans.push_back(dual_trans);
      Eigen::Vector3d dual_rot(d[3], d[4], d[5]);
      duals_rot.push_back(dual_rot);
    }
    ceres::CostFunction* translation_consensus =
        new cost_functions::DecentralEuclideanConsensus<3>(lambda_translation,
                                                           duals_trans);
    const auto trans_resid_id = ceres_problem_->AddResidualBlock(
        translation_consensus, NULL, frame_ptr->p_W_C_.data());
    consensus_resids_.insert(trans_resid_id);
    ceres::CostFunction* rotation_consensus =
        new cost_functions::DecentralRotationConsensus(
            lambda_rotation, frame_ptr->getReferenceRotation(), duals_rot);
    const auto rot_resid_id = ceres_problem_->AddResidualBlock(
        rotation_consensus, NULL, frame_ptr->q_W_C_.coeffs().data());
    consensus_resids_.insert(rot_resid_id);
    if (cam_type == Camera::Type::kPinholeSimple) {
      VectorOfVectorN<SimplePinholeCamera::kNumParameters> duals_intr;
      for (const auto& d : duals_frame_async) {
        Eigen::Matrix<double, SimplePinholeCamera::kNumParameters, 1> dual_intr;
        for (size_t i = 0; i < SimplePinholeCamera::kNumParameters; ++i) {
          dual_intr(i, 0) = d[6 + i];
        }
        duals_intr.push_back(dual_intr);
      }
      ceres::CostFunction* intrinsics_consensus =
          new cost_functions::DecentralEuclideanConsensus<
              SimplePinholeCamera::kNumParameters>(lambda_intrinsics,
                                                   duals_intr);
      const auto intr_resid_id = ceres_problem_->AddResidualBlock(
          intrinsics_consensus, NULL, frame_ptr->intrinsics_.data());
      consensus_resids_.insert(intr_resid_id);
    } else if (cam_type == Camera::Type::kPinhole) {
      VectorOfVectorN<PinholeCamera::kNumParameters> duals_intr;
      for (const auto& d : duals_frame_async) {
        Eigen::Matrix<double, PinholeCamera::kNumParameters, 1> dual_intr;
        for (size_t i = 0; i < PinholeCamera::kNumParameters; ++i) {
          dual_intr(i, 0) = d[6 + i];
        }
        duals_intr.push_back(dual_intr);
      }
      ceres::CostFunction* intrinsics_consensus =
          new cost_functions::DecentralEuclideanConsensus<
              PinholeCamera::kNumParameters>(lambda_intrinsics, duals_intr);
      const auto intr_resid_id = ceres_problem_->AddResidualBlock(
          intrinsics_consensus, NULL, frame_ptr->intrinsics_.data());
      consensus_resids_.insert(intr_resid_id);
    }
    switch (dist_type) {
      case Distortion::Type::kRadDist: {
        VectorOfVectorN<RadDistortion::kNumParameters> duals_dist;
        for (const auto& d : duals_frame_async) {
          Eigen::Matrix<double, RadDistortion::kNumParameters, 1> dual_dist;
          for (size_t i = 0; i < RadDistortion::kNumParameters; ++i) {
            dual_dist(i, 0) = d[6 + kNumIntrinsicParams + i];
          }
          duals_dist.push_back(dual_dist);
        }
        ceres::CostFunction* dist_consensus =
            new cost_functions::DecentralEuclideanConsensus<
                RadDistortion::kNumParameters>(lambda_distortion, duals_dist);
        const auto dist_resid_id = ceres_problem_->AddResidualBlock(
            dist_consensus, NULL, frame_ptr->dist_coeffs_.data());
        consensus_resids_.insert(dist_resid_id);
        break;
      }
      case Distortion::Type::kRadTanDist: {
        VectorOfVectorN<RadTanDistortion::kNumParameters> duals_dist;
        for (const auto& d : duals_frame_async) {
          Eigen::Matrix<double, RadTanDistortion::kNumParameters, 1> dual_dist;
          for (size_t i = 0; i < RadTanDistortion::kNumParameters; ++i) {
            dual_dist(i, 0) = d[6 + kNumIntrinsicParams + i];
          }
          duals_dist.push_back(dual_dist);
        }
        ceres::CostFunction* dist_consensus =
            new cost_functions::DecentralEuclideanConsensus<
                RadTanDistortion::kNumParameters>(lambda_distortion,
                                                  duals_dist);
        const auto dist_resid_id = ceres_problem_->AddResidualBlock(
            dist_consensus, NULL, frame_ptr->dist_coeffs_.data());
        consensus_resids_.insert(dist_resid_id);
        break;
      }
      case Distortion::Type::kEquiDist: {
        VectorOfVectorN<EquidistantDistortion::kNumParameters> duals_dist;
        for (const auto& d : duals_frame_async) {
          Eigen::Matrix<double, EquidistantDistortion::kNumParameters, 1>
              dual_dist;
          for (size_t i = 0; i < EquidistantDistortion::kNumParameters; ++i) {
            dual_dist(i, 0) = d[6 + kNumIntrinsicParams + i];
          }
          duals_dist.push_back(dual_dist);
        }
        ceres::CostFunction* dist_consensus =
            new cost_functions::DecentralEuclideanConsensus<
                EquidistantDistortion::kNumParameters>(lambda_distortion,
                                                       duals_dist);
        const auto dist_resid_id = ceres_problem_->AddResidualBlock(
            dist_consensus, NULL, frame_ptr->dist_coeffs_.data());
        consensus_resids_.insert(dist_resid_id);
        break;
      }
    }
  }
}

/// @brief compute residuals for all constraints
/// @return vector of residual values for the problem
auto Optimization::computeErrors() -> std::vector<double> {
  std::vector<double> result;
  if (ceres_problem_ == nullptr) {
    LOG(WARNING) << "Ceres-problem is empty";
    return result;
  }
  const auto frame_ids = data_ptr_->getFrameIds();
  if (frame_ids.empty()) {
    LOG(WARNING) << "No frames in ceres-problem";
    return result;
  }
  const auto map_point_ids = data_ptr_->getMapPointIds();
  if (map_point_ids.empty()) {
    LOG(WARNING) << "No map points in ceres-problem";
    return result;
  }
  for (const auto& id : consensus_resids_)
    ceres_problem_->RemoveResidualBlock(id);
  consensus_resids_.clear();

  ceres::Problem::EvaluateOptions eval_opts;
  eval_opts.apply_loss_function = false;
  double cost = 0;
  ceres_problem_->Evaluate(eval_opts, &cost, &result, nullptr, nullptr);
  return result;
}

/// @brief check the residual of a desired frame
/// @details retrieve frame observations and check if empty, and if not,
/// retrieve the camera. for each observation, retrieve the MapPoints and
/// calculate 3D MapPoint position in camera frame and project this point
/// in the image frame, then calculate the resulting residual!
/// @param frame_id frame to check
/// @return 
auto Optimization::checkFrame(const uint64_t frame_id) -> double {
  auto frame_ptr = data_ptr_->getFrame(frame_id);
  CHECK(frame_ptr != nullptr);
  const auto& observations = frame_ptr->getAllObservations();
  CHECK(!observations.empty());
  auto camera_ptr = frame_ptr->getCamera();
  std::vector<double> vals;
  for (const auto obs_i : observations) {
    auto map_point_ptr = data_ptr_->getMapPoint(obs_i.mp_id);
    CHECK(map_point_ptr != nullptr);
    CHECK(obs_i.frame_id == frame_id);
    const Eigen::Vector3d point_in_C =
        frame_ptr->q_W_C_.inverse() *
        (map_point_ptr->position_ - frame_ptr->p_W_C_);
    Eigen::Vector2d proj;
    auto proj_res = camera_ptr->projectPointUsingExternalParameters(
        frame_ptr->intrinsics_, frame_ptr->dist_coeffs_, point_in_C, &proj,
        nullptr, nullptr, nullptr);
    double repr_err = (proj - obs_i.obs).norm();
    vals.push_back(repr_err);
  }
  std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
  return vals[vals.size() / 2];
}

}  // namespace dba
