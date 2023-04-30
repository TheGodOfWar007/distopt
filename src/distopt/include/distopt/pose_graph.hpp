#pragma once

#include <memory>
#include <Eigen/Geometry>
#include <unordered_map>
#include <unordered_set>

#include <rapidcsv/rapidcsv.h>
#include <distopt/common.hpp>

// TypeDefs for common use across the library
using Timestamp = uint64_t; // timestamp in nanoseconds
using AgentID = uint64_t; // agent id
using NodeKey = std::pair<AgentID, Timestamp>; // key for a node in the pose graph. Strictly (agent_id, timestamp)
using GraphEdge = std::tuple<NodeKey, NodeKey>; // edge in the pose graph. Strictly (node1_key, node2_key)
using ResidualKey = std::pair<NodeKey, NodeKey>; // key for a residual in the pose graph. Strictly (node1_key, node2_key)
using RPStamp = std::pair<Timestamp, Timestamp>; // key for a relative pose measurement. Strictly (timestamp1, timestamp2)

namespace dba {

    // TypeDefs in namespace dba
    using Pose = PoseDual<kNumIntrinsicParams, kNumDistortionParams>;
    using PoseDelta = PoseDual<kNumIntrinsicParams, kNumDistortionParams>;
    using Residual = PoseDual<kNumIntrinsicParams, kNumDistortionParams>;

    using PosePtr = std::shared_ptr<Pose>;
    using PoseDeltaPtr = std::shared_ptr<PoseDelta>;
    using ResidualPtr = std::shared_ptr<Residual>;

    /// @brief A class to represent a relative pose measurement between two nodes (poses)
    /// A measurements is always between two pose nodes and corresponding to an edge in the
    /// pose graph.
    struct RPMeasurement {

        RPMeasurement() = delete;

        RPMeasurement(const RPStamp& stamp_pair, const GraphEdge& edge, const PoseDelta& pose_delta) : stamp_pair_(stamp_pair), edge_(edge), pose_delta_(pose_delta) {};

        auto getStampPair() -> RPStamp { return stamp_pair_; }

        auto getPoseDelta() -> PoseDelta { return pose_delta_; }

        auto getEdge() -> GraphEdge { return edge_; }

        private:
            RPStamp stamp_pair_; // timestamp pair
            GraphEdge edge_; // edge in the pose graph
            PoseDelta pose_delta_; // relative pose measurement
    };

    using RPMeasurementPtr = std::shared_ptr<RPMeasurement>;

    struct GraphNode {

        GraphNode() = delete;

        GraphNode(const Timestamp& timestamp, const AgentID& agent_id, const Pose& pose) : timestamp(timestamp), agent_id(agent_id), pose(pose), key_(agent_id, timestamp) {};

        auto getNeighbors() -> std::unordered_map<NodeKey, GraphNodePtr> { return neighbors_; }

        /// @brief Adds another node to the list of neighbors of the current node. 
        /// If the node is not from the same agent, then a corresponding residual is added.
        /// @param key key of the node to be added as a neighbor
        /// @param node pointer to the node to be added as a neighbor
        auto addNeighbor(const NodeKey& key, const GraphNodePtr& node) -> void { 
            neighbors_.insert(std::make_pair(key, node)); 
            neighbor_keys_.push_back(key);
            neighbor_nodes_.push_back(node);

            if(node->agent_id != agent_id){
                has_residuals = true;
                ResidualKey residual_key = std::make_pair(key_, key);
                Residual residual;
                residual_keys_.push_back(residual_key);
                residuals_.emplace(residual_key, std::make_shared<Residual>(residual));
                residual_list_.push_back(residual);
                residual_parents_.push_back(key);
            }
        }

        auto removeNeighbor(const NodeKey& key) -> void { 
            neighbors_.erase(key); 
            neighbor_keys_.erase(std::remove(neighbor_keys_.begin(), neighbor_keys_.end(), key), neighbor_keys_.end());
            neighbor_nodes_.erase(std::remove(neighbor_nodes_.begin(), neighbor_nodes_.end(), neighbors_.at(key)), neighbor_nodes_.end());

            if(neighbors_.at(key)->agent_id != agent_id){
                ResidualKey residual_key = std::make_pair(key_, key);
                residuals_.erase(residual_key);
                residual_keys_.erase(std::remove(residual_keys_.begin(), residual_keys_.end(), residual_key), residual_keys_.end());
                residual_list_.erase(std::remove(residual_list_.begin(), residual_list_.end(), residuals_.at(residual_key)), residual_list_.end());
                residual_parents_.erase(std::remove(residual_parents_.begin(), residual_parents_.end(), key), residual_parents_.end());
                
                if(residual_parents_.empty()){
                    has_residuals = false;
                }
            }
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

        auto getResidualKeys() -> std::vector<ResidualKey> { return residual_keys_; }

        auto getResiduals() -> std::unordered_map<ResidualKey, ResidualPtr> { return residuals_; }

        auto getResidualList() -> std::vector<Residual> { return residual_list_; }

        auto getResidualParents() -> std::vector<NodeKey> { return residual_parents_; }

        auto getNumResiduals() -> size_t { return residual_keys_.size(); }

        auto getNumNeighbors() -> size_t { return neighbor_keys_.size(); }

        auto getNumResidualParents() -> size_t { return residual_parents_.size(); }

        auto getNumForeignNeighbors() -> size_t { 
            return getNumResidualParents();
        }

        private:
            NodeKey key_; // key of the node
            std::unordered_map<NodeKey, GraphNodePtr> neighbors_;  // map of neighbor keys to neighbor pointers
            std::vector<NodeKey> neighbor_keys_; // list of neighbor keys
            std::vector<GraphNodePtr> neighbor_nodes_; // list of neighbor pointers
            Timestamp timestamp; // timestamp of the node
            AgentID agent_id; // agent id of the node
            Pose pose; // pose of the node
            bool has_residuals; // flag to indicate if the node has residuals
            std::vector<NodeKey> residual_parents_; // list of agent id pairs for residuals.
            std::unordered_map<ResidualKey, ResidualPtr> residuals_;
            std::vector<Residual> residual_list_; // list of residuals
            std::vector<ResidualKey> residual_keys_; // list of residual keys
    };

    using GraphNodePtr = std::shared_ptr<GraphNode>;

    struct PoseGraph {

        PoseGraph() = delete;

        private:
            std::unordered_map<NodeKey, GraphNodePtr> node_ptrs_; // map of node keys to node pointers
            std::vector<GraphEdge> edges_; // list of edges
            std::vector<NodeKey> node_keys_; // list of node keys
            std::vector<GraphNodePtr> node_ptr_list_; // list of node pointers
            std::unordered_map<GraphEdge, RPMeasurementPtr> measurements_; // map of edges to relative pose measurements
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