#include <gflags/gflags.h>
#include <glog/logging.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <Eigen>

#include "rclcpp/rclcpp.hpp"

// uncomment once we've written our async comm file
#include "distopt/asynchronous_communication.hpp"
#include "distopt/common.hpp"
#include "distopt/data.hpp"
#include "distopt/distortion.hpp"
#include "distopt/optimization.hpp"

#include "robopt_open/local-parameterization/pose-quaternion-local-param.h"
#include "robopt_open/posegraph-error/six-dof-between.h"

DECLARE_string(admm_type);
DECLARE_string(data_folder);
DECLARE_string(result_folder);
DECLARE_uint64(num_admm_iter);

using namespace dba;

/// @brief AgentNode that will have a problem and local beliefs and will communicate with others
class AgentNode : public rclcpp::Node
{
    public:
        AgentNode() : Node("agent_node"), counter_(0)
        {
            rclcpp::Rate loop_rate(100);                // execute loop at 100Hz
            neighbor_ids_ = data_ptr_->getNeighbors();
            // do stuff in here

            // TODO: rank is presumably node number, in DBA 0 is held for a coordinator and storage node

            // if (rank == 0) {
            //     // This is the coordinator and storage node
            //     const size_t num_nodes = world_size - 1;
            //     std::vector<dba::DataSharedPtr> data(num_nodes);
            //     for (size_t i = 0; i < num_nodes; ++i) {
            //     data[i] = std::make_shared<dba::Data>(i);
            //     CHECK(data[i]->readDataFromFiles(FLAGS_data_folder));
            //     }
            //     dba::AsynchronousCoordinator coordinator(data);
            // } else {
            //     // These are the actual worker nodes
            //     dba::DataSharedPtr data_ptr(new dba::Data(rank - 1));
            //     dba::DataSharedPtr data_copy_ptr(new dba::Data(rank - 1));
            //     CHECK(data_ptr->readDataFromFiles(FLAGS_data_folder));
            //     CHECK(data_copy_ptr->readDataFromFiles(FLAGS_data_folder));
            //     dba::AsynchronousCommunication node(data_ptr, data_copy_ptr);
            // }
            
            // need to define: translations, rotations, and measurements from data set send them here
            // ceres::Problem* problem = createPoseGraph();

            while (rclcpp::ok()) {
                // execute main loop!
                execute();
            }
        }
        auto communicateData() -> void;
        

    private:
        /// @brief execute optimization loop (?)
        void execute() {

        }
        
        int counter_;
        int flag_;
        int finish_flag_;

        size_t idx;
        Eigen::Quaterniond orientation;
        Eigen::Vector3d position;
        std::vector<uint64_t> neighbor_ids_;

        std::vector<Eigen::Quaterniond> rotations;
        std::vector<Eigen::Vector3d> translations;
        std::vector<PoseConnection> measurements;

        DataSharedPtr data_ptr_;

};

/// @brief pose connection struct between two agents
/// @details describes delta rotation, delta translation, and covariance of pose from agent 1 to agent 2
struct PoseConnection {
    size_t idx_pose_1;
    size_t idx_pose_2;
    Eigen::Quaterniond delta_rotation;
    Eigen::Vector3d delta_translation;
    Eigen::Matrix<double, 6, 6> covariance;
};

double* createCeresParameterForPose(const Eigen::Quaterniond& q, const Eigen::Vector3d& t) {
    double* pose_parameter = new double[7];
    for (size_t i = 0; i < 4; ++i) 
        pose_parameter[i] = q.coeffs().data()[i];
    for (size_t i = 0; i < 3; ++i) 
        pose_parameter[i] = t[i];    
    return pose_parameter;
}

/// @brief create pose graph problem
/// @param rotations vector of quaternions representing rotations
/// @param translations vector of positions representing translations
/// @param measurements pose connection between two agents
/// @return ceres problem for the given pose and agent information
ceres::Problem* createPoseGraph(const std::vector<Eigen::Quaterniond>& rotations, 
                                const std::vector<Eigen::Vector3d>& translations, 
                                const std::vector<PoseConnection>& measurements) {
    // Create a ceres problem (this will be filled later)
    ceres::Problem::Options prob_opts;
    prob_opts.enable_fast_removal = true;
    ceres::Problem* problem = new ceres::Problem(prob_opts);
    
    // Create local parameterization
    ceres::LocalParameterization* local_param = new robopt::local_param::PoseQuaternionLocalParameterization();
    
    // Create a dummy parameter block for identity pose (we need two since ceres doesn't accept the same pointer twice within a cost function)
    double* identity_pose1 = createCeresParameterForPose(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
    problem->AddParameterBlock(identity_pose1, robopt::defs::pose::kPoseBlockSize, local_param);
    double* identity_pose2 = createCeresParameterForPose(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
    problem->AddParameterBlock(identity_pose2, robopt::defs::pose::kPoseBlockSize, local_param);
    
    // Add the parameter blocks
    // const size_t num_poses = poses.size();       // poses not defined but should just be the size of rotations...
    const size_t num_poses = rotations.size();
    std::vector<double*> pose_parameters(num_poses);
    for (size_t pose_idx = 0; pose_idx < num_poses; ++pose_idx) {
        const auto& q_w_s = rotations[pose_idx];
        const auto& t_w_s = translations[pose_idx];
        double* pose_parameter = createCeresParameterForPose(q_w_s, t_w_s);
        pose_parameters[pose_idx] = pose_parameter;
    } 
    
    // Add the residuals
    for (const auto& meas : measurements) {
        // Compute the square root information matrix
        Eigen::Matrix<double, 6, 6> sqrt_info;
        Eigen::Matrix<double, 6, 6> L = meas.covariance.llt().matrixL();
        L.template triangularView<Eigen::Lower>().solveInPlace(sqrt_info);
        
        // Create and add the residual to the problem
        ceres::CostFunction* cost = new robopt::posegraph::SixDofBetweenError(meas.delta_rotation, meas.delta_translation, sqrt_info, robopt::defs::pose::PoseErrorType::kImu);
        problem->AddResidualBlock(cost, NULL, pose_parameters[meas.idx_pose_1], pose_parameters[meas.idx_pose_2], identity_pose1, identity_pose2);
    }
    
    return problem;
}

auto AgentNode::communicateData() -> void {
    const size_t num_neighbors = neighbor_ids_.size() - 1;


}

int main(int argc, char** argv) {
    // google::InitGoogleLogging(argv[0]);
    // gflags::ParseCommandLineFlags(&argc, &argv, true);
    // CHECK(FLAGS_admm_type == "decentral_async");

    

    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AgentNode>());
    rclcpp::shutdown();
    return 0;
}