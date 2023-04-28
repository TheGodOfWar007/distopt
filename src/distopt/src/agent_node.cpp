#include <gflags/gflags.h>
#include <glog/logging.h>
#include <chrono>
#include <fstream>
#include <iostream>

#include "rclcpp/rclcpp.hpp"

// uncomment once we've written our async comm file
#include "distopt/asynchronous_communication.hpp"
#include "distopt/common.hpp"
#include "distopt/data.hpp"
#include "distopt/distortion.hpp"
#include "distopt/optimization.hpp"

DECLARE_string(admm_type);
DECLARE_string(data_folder);
DECLARE_string(result_folder);
DECLARE_uint64(num_admm_iter);

using namespace dba;

class AgentNode : public rclcpp::Node
{
    public:
        AgentNode() : Node("agent_node"), counter_(0)
        {
            neighbor_ids_ = data_ptr_->getNeighbors();
            // do stuff in here
        }

    private:
        void communicateData() {
            const size_t num_neighbors = neighbor_ids_.size() - 1;


        }

        // NOTE: rather than having a header file, define variables in private as is usually done for ROS nodes it seems?
        int counter_;
        int flag_;
        int finish_flag_;
        std::vector<uint64_t> neighbor_ids_;
        DataSharedPtr data_ptr_;

};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AgentNode>());
    rclcpp::shutdown();
    return 0;
}