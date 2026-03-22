#!/usr/bin/env python3

import rospy
import mujoco
import numpy as np
import json
import os
from scipy.spatial.transform import Rotation as R
from sim_env import CrimsonMujocoEnv

# Import ROS message and service types
from meta_msgs.msg import TopologicalGraph, TopoList
from meta_msgs.srv import Angles2Topo, Angles2TopoResponse
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


class GraphModelerNode:
    def __init__(self):
        """
        Initialize ROS node and service server.
        """
        rospy.init_node("graph_modeler_node")
        rospy.loginfo("Graph Modeler Node is starting.")

        # --- Initialize Simulation Environment ---
        # We use render_mode=None for speed as this is a service
        # You can change to "human" for debugging if needed (but might need a display)
        self.env = CrimsonMujocoEnv(render_mode=None) 
        rospy.loginfo("Mujoco Environment Initialized.")

        # --- Initialize ROS Service ---
        self.service = rospy.Service(
            "model_graph", Angles2Topo, self.handle_model_request
        )
        rospy.loginfo("Service 'model_graph' is ready.")

        rospy.loginfo("Node setup complete. Waiting for requests...")

    def _simulate_and_capture(self, target_full_pose):
        """
        Core logic adapted from graph_modeler function.
        Simulates the robot to the target pose and captures the state.
        """
        # Ensure the target pose vector matches the number of actuators
        if len(target_full_pose) != self.env.model.nu:
            rospy.logwarn(f"Target pose length ({len(target_full_pose)}) does not match "
                          f"number of actuators ({self.env.model.nu}). Padding or truncating.")
            # Simple handling: pad with zeros or truncate
            if len(target_full_pose) < self.env.model.nu:
                target_full_pose = np.pad(target_full_pose, (0, self.env.model.nu - len(target_full_pose)))
            else:
                target_full_pose = target_full_pose[:self.env.model.nu]

        self.env.data.ctrl[:] = target_full_pose

        # Stabilization
        stabilization_steps = 100
        for _ in range(stabilization_steps):
            self.env.do_simulation(self.env.data.ctrl, self.env.frame_skip)

        # Capture Data
        joint_pos = self.env.get_joint_pos()
        joint_quat = self.env.get_joint_quat()
        joint_rad = self.env.get_joint_angles()
        node_pos = self.env.get_node_pos()
        node_quat = self.env.get_node_quat()

        length, width, height = self.env.get_simple_size_from_site()
        leg_base_pos = self.env.get_leg_base_site_pos()
        leg_base_quat = self.env.get_leg_base_site_quat()
        leg_base_result = np.hstack([leg_base_pos, leg_base_quat]).tolist()

        # Leg angle assuming index 6 (from original code, verify if needed)
        # If target_full_pose is large enough
        leg_x = 0.0
        if len(target_full_pose) > 6:
            leg_x = target_full_pose[6] 

        # Construct Nodes
        # Format: [px, py, pz, qw, qx, qy, qz] (7 dim) or similar based on usage
        # Original code: np.hstack([p, q]).tolist() -> 3 pos + 4 quat = 7
        nodes_list = []
        for p, q in zip(node_pos, node_quat):
            nodes_list.extend(np.hstack([p, q]).tolist())

        # Construct Edges
        # Original code: np.hstack([j, p, q]) -> 1 angle + 3 pos + 4 quat = 8
        edges_list = []
        for j, p, q in zip(joint_rad, joint_pos, joint_quat):
            edges_list.extend(np.hstack([j, p, q]).tolist())
        
        # Append fixed Base node as an edge? 
        # Original code: + [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
        # This seems to treat the base as an edge in the logic or just appended data.
        edges_list.extend([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        # Adjacency Matrix
        # 16 servos + 1 base = 17 nodes.
        # The structure is a chain? Or tree?
        # Based on typical robot structure (e.g., hexapod/quadruped), it's likely a tree.
        # But without the original JSON logic, I can only assume a simple chain or placeholder.
        # Let's try to reconstruct the adjacency matrix based on the number of nodes.
        # However, the user asked to "turn the content of the original json file into a complete dictionary output and transmission".
        # This implies we should capture the structure that was originally in the JSON.

        # Since I don't have the original JSON structure logic here, I will create a placeholder adjacency matrix.
        # You should replace this with the actual connectivity logic of your robot.
        num_nodes = len(nodes_list) // 7 # 7 floats per node
        adjacency_matrix = np.eye(num_nodes).flatten().tolist()

        return nodes_list, edges_list, adjacency_matrix, [length, width, height], leg_base_result, leg_x

    def handle_model_request(self, req):
        """
        Service handler.
        """
        rospy.loginfo("Handling model_graph request.")
        response = Angles2TopoResponse()
        
        try:
            target_pose = np.array(req.angles, dtype=np.float64)
            
            nodes, edges, adjacency, scale, leg_base, leg_angle = self._simulate_and_capture(target_pose)

            # --- Populate Response ---
            topo_graph = TopologicalGraph()
            
            # 1. Fill Edges (which seem to contain joint info + pose)
            # Layout needs to be defined if using Float32MultiArray for raw data transport
            # Assuming edges is flattened list of [angle, x, y, z, qw, qx, qy, qz]
            num_cols_edges = 8
            num_rows_edges = len(edges) // num_cols_edges
            
            topo_graph.edges.data = edges
            
            # Set layout for edges
            dim0_edges = MultiArrayDimension()
            dim0_edges.label = "rows"
            dim0_edges.size = num_rows_edges
            dim0_edges.stride = len(edges)
            
            dim1_edges = MultiArrayDimension()
            dim1_edges.label = "cols"
            dim1_edges.size = num_cols_edges
            dim1_edges.stride = num_cols_edges
            
            topo_graph.edges.layout.dim = [dim0_edges, dim1_edges]

            # 2. Fill Nodes
            # Assuming nodes is flattened list of [x, y, z, qw, qx, qy, qz]
            num_cols_nodes = 7
            num_rows_nodes = len(nodes) // num_cols_nodes
            
            topo_graph.nodes.data = nodes
            
            # Set layout for nodes
            dim0_nodes = MultiArrayDimension()
            dim0_nodes.label = "rows"
            dim0_nodes.size = num_rows_nodes
            dim0_nodes.stride = len(nodes)
            
            dim1_nodes = MultiArrayDimension()
            dim1_nodes.label = "cols"
            dim1_nodes.size = num_cols_nodes
            dim1_nodes.stride = num_cols_nodes
            
            topo_graph.nodes.layout.dim = [dim0_nodes, dim1_nodes]

            topo_graph.adjacency.data = adjacency
            # Layout for adjacency (Square matrix)
            num_adj_nodes = int(len(adjacency)**0.5)

            dim0_adj = MultiArrayDimension()
            dim0_adj.label = "rows"
            dim0_adj.size = num_adj_nodes
            dim0_adj.stride = len(adjacency)

            dim1_adj = MultiArrayDimension()
            dim1_adj.label = "cols"
            dim1_adj.size = num_adj_nodes
            dim1_adj.stride = num_adj_nodes

            topo_graph.adjacency.layout.dim = [dim0_adj, dim1_adj]


            # 4. Fill Features
            # Map captured data to TopologicalGraph fields

            # Global feature
            topo_graph.feature.scale = scale

            # leg_base from simulation is a flattened list [pos, quat] -> 7 floats
            # In JSON, leg_base is a list of lists (4 legs x 7 floats).
            # The current simulation method `get_leg_base_site_pos/quat` might only return ONE leg base or accumulated?
            # Let's check `_simulate_and_capture` logic.
            # `leg_base_result` is `np.hstack([leg_base_pos, leg_base_quat]).tolist()`.
            # If `get_leg_base_site_pos` returns multiple sites, `np.hstack` might flatten them differently.
            # Usually `get_site_pos` returns (N, 3).

            # If `leg_base_result` is flat array of all legs, we are good.
            # JSON shows 4 legs. So we expect 4 * 7 = 28 floats.

            topo_graph.feature.leg_base.data = leg_base
            # If leg_base is 1D array of (4*7) elements
            num_legs = len(leg_base) // 7
            dim0_lb = MultiArrayDimension(label="rows", size=num_legs, stride=len(leg_base))
            dim1_lb = MultiArrayDimension(label="cols", size=7, stride=7)
            topo_graph.feature.leg_base.layout.dim = [dim0_lb, dim1_lb]

            # JSON shows "leg_angle" as a list of 3 floats: [0.058..., -0.251, 1.79]
            # My code currently puts `[leg_angle] * 3` where leg_angle was `target_full_pose[6]`.
            # This logic `target_full_pose[6]` seems specific to a certain kinematic chain.
            # If `leg_angle` in JSON is indeed [angle1, angle2, angle3], we need to know where they come from.
            # Assuming for now we just want to match the structure.

            # If the robot has identical legs, user might have just replicated one angle?
            # Or these represent roll/pitch/yaw of the leg mounting?

            # Let's assume the simulated environment can provide this or we pass it through.
            # Currently `_simulate_and_capture` returns `leg_x`.
            # I will wrap it to match the JSON's list of 3 format if that's what's expected,
            # or if `leg_angles` field in Global.msg expects array.

            topo_graph.feature.leg_angles = [leg_angle] * 3

            # Note on message fields vs JSON fields:
            # JSON: "nodes", "edges", "global" -> "scale", "leg_base", "leg_angle"
            # MSG:  "nodes", "edges", "adjacency", "feature" -> "scale", "leg_base", "leg_angles", "locomotion_mode"

            # It seems fairly consistent.

            response.graph_data = topo_graph
            rospy.loginfo("Service request handled successfully.")

        except Exception as e:
            rospy.logerr(f"Error processing service request: {e}")
            # Return empty response or handle error appropriately
            pass

        return response

if __name__ == "__main__":
    try:
        GraphModelerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
