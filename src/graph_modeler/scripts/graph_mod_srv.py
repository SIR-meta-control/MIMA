#!/usr/bin/env python3

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rospy
import numpy as np
from sim_env import CrimsonMujocoEnv

# Import ROS message and service types
from meta_msgs.msg import TopologicalGraph
from meta_msgs.srv import Angles2Topo, Angles2TopoResponse
from std_msgs.msg import MultiArrayDimension


class GraphModelerNode:
    def __init__(self):
        """
        Initialize ROS node and service server.
        """
        rospy.init_node("graph_modeler_node")
        rospy.loginfo("Graph Modeler Node is starting.")

        # Capture service requests without rendering.
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
        target_full_pose = np.asarray(target_full_pose, dtype=np.float64)
        if target_full_pose.ndim != 1 or not np.isfinite(target_full_pose).all():
            raise ValueError(
                "Target pose must be a finite one-dimensional angle vector"
            )
        # Ensure the target pose vector matches the number of actuators
        if len(target_full_pose) != self.env.model.nu:
            rospy.logwarn(
                f"Target pose length ({len(target_full_pose)}) does not match "
                f"number of actuators ({self.env.model.nu}). Padding or truncating."
            )
            # Simple handling: pad with zeros or truncate
            if len(target_full_pose) < self.env.model.nu:
                target_full_pose = np.pad(
                    target_full_pose, (0, self.env.model.nu - len(target_full_pose))
                )
            else:
                target_full_pose = target_full_pose[: self.env.model.nu]

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
        leg_base_result = np.hstack([leg_base_pos, leg_base_quat])
        if leg_base_result.shape != (4, 7) or not np.isfinite(leg_base_result).all():
            raise ValueError("Leg-base poses must be a finite 4-by-7 matrix")
        leg_base_result = leg_base_result.reshape(-1).tolist()

        # Legacy limb-angle capture uses joint index 6.
        leg_x = 0.0
        if len(target_full_pose) > 6:
            leg_x = target_full_pose[6]

        # Node rows: [x, y, z, qw, qx, qy, qz].
        nodes_list = []
        for p, q in zip(node_pos, node_quat):
            nodes_list.extend(np.hstack([p, q]).tolist())

        # Edge rows: [angle, x, y, z, qw, qx, qy, qz].
        edges_list = []
        for j, p, q in zip(joint_rad, joint_pos, joint_quat):
            edges_list.extend(np.hstack([j, p, q]).tolist())

        # Append the fixed base reference pose.
        edges_list.extend([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        # Identity adjacency; physical link connectivity is not inferred here.
        num_nodes = len(nodes_list) // 7  # 7 floats per node
        adjacency_matrix = np.eye(num_nodes).flatten().tolist()

        return (
            nodes_list,
            edges_list,
            adjacency_matrix,
            [length, width, height],
            leg_base_result,
            leg_x,
        )

    def handle_model_request(self, req):
        """
        Service handler.
        """
        rospy.loginfo("Handling model_graph request.")
        response = Angles2TopoResponse()

        try:
            target_pose = np.array(req.angles, dtype=np.float64)

            nodes, edges, adjacency, scale, leg_base, leg_angle = (
                self._simulate_and_capture(target_pose)
            )

            # --- Populate Response ---
            topo_graph = TopologicalGraph()

            # Pack eight-column joint data.
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

            # Pack seven-column body poses.
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
            num_adj_nodes = int(len(adjacency) ** 0.5)

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

            # Copy the leg-base poses returned by the simulation accessor.
            topo_graph.feature.leg_base.data = leg_base
            # If leg_base is 1D array of (4*7) elements
            num_legs = len(leg_base) // 7
            dim0_lb = MultiArrayDimension(
                label="rows", size=num_legs, stride=len(leg_base)
            )
            dim1_lb = MultiArrayDimension(label="cols", size=7, stride=7)
            topo_graph.feature.leg_base.layout.dim = [dim0_lb, dim1_lb]

            # Legacy field repeats one joint angle in all three entries.
            topo_graph.feature.leg_angles = [leg_angle] * 3

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
