#pragma once

#include <memory>
#include <Eigen/Geometry>
#include <unordered_map>
#include <unordered_set>

#include <rapidcsv/rapidcsv.h>
#include <distopt/common.hpp>

// TypeDefs for common use across the library
using Timestamp = uint64_t;
using AgentID = uint64_t;
using NodeKey = std::pair<AgentID, Timestamp>;
using ResidualKey = std::pair<NodeKey, NodeKey>;
using RPStamp = std::pair<Timestamp, Timestamp>;

namespace dba {

    // TypeDefs in namespace dba
    using Pose = PoseDual<kNumIntrinsicParams, kNumDistortionParams>;
    using PoseDelta = PoseDual<kNumIntrinsicParams, kNumDistortionParams>;
    using Residual = PoseDual<kNumIntrinsicParams, kNumDistortionParams>;

    /// @brief A class to represent a relative pose measurement between two nodes (poses)
    struct RPMeasurement {

        RPMeasurement(const RPStamp& stamp_pai, const PoseDelta& pose_delta) : stamp_pair_(stamp_pair_), pose_delta_(pose_delta) {};

        auto getStampPair() -> RPStamp { return stamp_pair_; }

        auto getPoseDelta() -> PoseDelta { return pose_delta_; }

        private:
            RPStamp stamp_pair_;
            PoseDelta pose_delta_;
    };

    using RPMeasurementPtr = std::shared_ptr<RPMeasurement>;

    struct GraphNode {
        // Timestamp timestamp;
        // AgentID agent_id;
        // Pose pose;

        GraphNode(const Timestamp& timestamp, const AgentID& agent_id, const Pose& pose) : timestamp(timestamp), agent_id(agent_id), pose(pose) {};

        auto getNeighbors() -> std::unordered_map<NodeKey, GraphNodePtr> { return neighbors_; }

        auto addNeighbor(const NodeKey& key, const GraphNodePtr& node) -> void { 
            neighbors_.insert(std::make_pair(key, node)); 
            neighbor_keys_.push_back(key);
            neighbor_nodes_.push_back(node);

            if(node->agent_id != agent_id){
                has_residuals = true;
            }
        }

        auto removeNeighbor(const NodeKey& key) -> void { 
            neighbors_.erase(key); 
            neighbor_keys_.erase(std::remove(neighbor_keys_.begin(), neighbor_keys_.end(), key), neighbor_keys_.end());
            neighbor_nodes_.erase(std::remove(neighbor_nodes_.begin(), neighbor_nodes_.end(), neighbors_.at(key)), neighbor_nodes_.end());
        }

        auto getNeighbor(const NodeKey& key) -> GraphNodePtr { return neighbors_.at(key); }

        auto isNeighbor(const NodeKey& key) -> bool { return neighbors_.find(key) != neighbors_.end(); }

        auto isNeighbor(const GraphNodePtr& node) -> bool { return std::find(neighbor_nodes_.begin(), neighbor_nodes_.end(), node) != neighbor_nodes_.end(); }

        auto getNeighborKeys() -> std::vector<NodeKey> { return neighbor_keys_; }

        auto getNeighborNodes() -> std::vector<GraphNodePtr> { return neighbor_nodes_; }

        auto hasNeighbors() -> bool { return !neighbors_.empty(); }

        auto getTimestamp() -> Timestamp { return timestamp; }

        auto getAgentID() -> AgentID { return agent_id; }

        auto getPose() -> Pose { return pose; }

        auto setPose(const Pose& pose) -> void { this->pose = pose; }

        auto hasResiduals() -> bool { return has_residuals; }

        private:
            std::unordered_map<NodeKey, GraphNodePtr> neighbors_; 
            std::vector<NodeKey> neighbor_keys_;
            std::vector<GraphNodePtr> neighbor_nodes_;
            Timestamp timestamp;
            AgentID agent_id;
            Pose pose;
            bool has_residuals;
            std::vector<std::pair<AgentID, AgentID>> residual_parents_;
            std::unordered_map<ResidualKey, Residual> residuals_;
            std::vector<Residual> residual_list_;
            std::vector<ResidualKey> residual_keys_;
    };

    using GraphNodePtr = std::shared_ptr<GraphNode>;

    struct PoseGraph {

        private:
            std::unordered_map<NodeKey, GraphNode> nodes_;
            std::vector<std::tuple<NodeKey, NodeKey>> edges_;
            std::vector<NodeKey> node_keys_;
            std::vector<GraphNodePtr> node_ptrs_;
            std::vector<RPMeasurementPtr> measurements_;
    };

    using PoseGraphPtr = std::shared_ptr<PoseGraph>;


    class PGFromCSV{
        /// @brief creates pose graph from CSV file
        /// @param path path to CSV file
        /// @return bool
        // auto PGFromCSV(const std::string& path) -> bool;

        private:
            /// @brief pose graph
            std::unordered_map<uint64_t, Pose> pose_graph_;
    };

} // namespace dba