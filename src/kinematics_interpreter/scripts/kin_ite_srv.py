#!/usr/bin/env python3

import yaml
import rospy
import mujoco
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# 导入ROS消息和服务类型
from meta_msgs.msg import TopologicalGraph
from meta_msgs.srv import Topo2Angles, Topo2AnglesResponse
from std_msgs.msg import Float32MultiArray


class KinematicsInterpreterNode:
    def __init__(self):
        """
        初始化ROS节点，加载YAML，并同时设置服务服务器和订阅/发布器。
        """
        rospy.init_node("kinematics_interpreter_node")
        rospy.loginfo("Kinematics Interpreter Node is starting.")

        self.flag_test = rospy.get_param("flag_test", True)

        # --- 1. 加载静态YAML文件 ---
        self.mujoco_relative_poses_pos = []
        self.mujoco_relative_poses_quat = []
        self.load_yaml_data()
        if not self.mujoco_relative_poses_pos:
            rospy.signal_shutdown("Failed to load YAML file. Shutting down.")
            return

        # --- 2. 初始化ROS服务 ---
        self.service = rospy.Service(
            "interpret_kinematics", Topo2Angles, self.handle_interpret_request
        )
        rospy.loginfo("Service server is ready.")

        # --- 3. 初始化用于处理 optimal_topo 的订阅者和发布器 ---
        # 订阅 "optimal topo" 指令
        self.topo_subscriber = rospy.Subscriber(
            "/optimal_topology",
            TopologicalGraph,  # 消息类型就是拓扑图
            self.optimal_topo_callback,  # 新的回调函数
            queue_size=10,
        )
        # 发布 "motor_ctrl" 指令
        self.motor_ctrl_publisher = rospy.Publisher(
            "/motor_ctrl",
            Float32MultiArray,  # 消息类型是一个浮点数组
            queue_size=10,
        )
        rospy.loginfo("Subscriber for optimal topology is ready.")
        rospy.loginfo("Node setup complete. Waiting for requests or messages...")

    def load_yaml_data(self):
        try:
            script_dir = Path(__file__).parent
            yaml_path = script_dir / "theoretically_relative_poses.yaml"
            rospy.loginfo(f"Attempting to load YAML file from: {yaml_path}")
            with open(yaml_path, "r") as yaml_file:
                data = yaml.safe_load(yaml_file)
            relative_poses_from_yaml = data["kinematic_chain_relative_poses"]
            for pose_data in relative_poses_from_yaml:
                pos = np.array(pose_data["position"], dtype=np.double)
                quat_wxyz = np.array(pose_data["quaternion_wxyz"], dtype=np.double)
                self.mujoco_relative_poses_pos.append(pos)
                self.mujoco_relative_poses_quat.append(quat_wxyz)
            rospy.loginfo(
                f"Successfully loaded {len(self.mujoco_relative_poses_pos)} relative poses."
            )
        except FileNotFoundError:
            rospy.logerr(f"Error: The YAML file '{yaml_path}' was not found.")
        except Exception as e:
            rospy.logerr(
                f"An error occurred while reading or parsing the YAML file: {e}"
            )

    def _calculate_control_vector(self, reference_edges_data, reference_leg_angel):
        """
        ## 重构 ##
        这是核心的计算逻辑，被提取出来作为一个独立的内部函数。
        它接收解析好的数据，返回计算出的控制向量列表。
        """
        try:
            num_json_edges = len(reference_edges_data)
            if num_json_edges < 2:
                rospy.logwarn("Not enough edges for calculation.")
                return None

            # --- 核心计算逻辑 ---
            num_transformations = min(
                num_json_edges - 1, len(self.mujoco_relative_poses_pos)
            )
            result_joint_angles_rad = []
            # 完整数据的开关
            Complete_data = self.flag_test

            for i in range(num_transformations):
                current_json_edge_data = reference_edges_data[i]
                if Complete_data:
                    json_edge_i_pos = np.array(
                        current_json_edge_data[1:4], dtype=np.double
                    )
                    json_edge_i_quat_wxyz = np.array(
                        current_json_edge_data[4:8], dtype=np.double
                    )
                else:
                    json_edge_i_pos = np.array(
                        current_json_edge_data[:3], dtype=np.double
                    )
                    json_edge_i_quat_wxyz = np.array(
                        current_json_edge_data[3:7], dtype=np.double
                    )
                if json_edge_i_quat_wxyz.shape[0] != 4:
                    continue
                mujoco_rel_pos = self.mujoco_relative_poses_pos[i]
                mujoco_rel_quat_wxyz = self.mujoco_relative_poses_quat[i]
                predicted_next_edge_pos = np.empty(3, dtype=np.double)
                predicted_next_edge_quat_wxyz = np.empty(4, dtype=np.double)
                mujoco.mju_mulPose(
                    predicted_next_edge_pos,
                    predicted_next_edge_quat_wxyz,
                    json_edge_i_pos,
                    json_edge_i_quat_wxyz,
                    mujoco_rel_pos,
                    mujoco_rel_quat_wxyz,
                )
                next_json_edge_data = reference_edges_data[i + 1]
                if Complete_data:
                    actual_next_json_edge_pos = np.array(
                        next_json_edge_data[1:4], dtype=np.double
                    )
                    actual_next_json_edge_quat_wxyz = np.array(
                        next_json_edge_data[4:8], dtype=np.double
                    )
                else:
                    actual_next_json_edge_pos = np.array(
                        next_json_edge_data[:3], dtype=np.double
                    )
                    actual_next_json_edge_quat_wxyz = np.array(
                        next_json_edge_data[3:7], dtype=np.double
                    )

                if actual_next_json_edge_quat_wxyz.shape[0] != 4:
                    continue
                q_diff_wxyz = np.empty(4, dtype=np.double)
                inv_predicted_next_edge_quat_wxyz = np.empty(4, dtype=np.double)
                mujoco.mju_negQuat(
                    inv_predicted_next_edge_quat_wxyz, predicted_next_edge_quat_wxyz
                )
                mujoco.mju_mulQuat(
                    q_diff_wxyz,
                    inv_predicted_next_edge_quat_wxyz,
                    actual_next_json_edge_quat_wxyz,
                )
                q_diff_xyzw = np.array(
                    [q_diff_wxyz[1], q_diff_wxyz[2], q_diff_wxyz[3], q_diff_wxyz[0]]
                )
                r_diff = R.from_quat(q_diff_xyzw)
                euler_angles_xyz_intrinsic = r_diff.as_euler("XYZ", degrees=True)
                joint_angle_from_deviation_deg = euler_angles_xyz_intrinsic[2]
                rad = np.radians(joint_angle_from_deviation_deg)
                result_joint_angles_rad.append(rad)
            rospy.loginfo("Finished calculating all joint angle deviations.")

            # --- 构造控制向量 ---
            control_vector_list = [
                -np.abs(result_joint_angles_rad[3]),
                result_joint_angles_rad[0],
                result_joint_angles_rad[2],
                -np.abs(result_joint_angles_rad[3]),
                result_joint_angles_rad[5],
                -reference_leg_angel[0],
                reference_leg_angel[1],
                reference_leg_angel[2],
                reference_leg_angel[0],
                reference_leg_angel[1],
                reference_leg_angel[2],
                -reference_leg_angel[0],
                reference_leg_angel[1],
                reference_leg_angel[2],
                reference_leg_angel[0],
                reference_leg_angel[1],
                reference_leg_angel[2],
            ]
            return control_vector_list

        except IndexError as e:
            rospy.logerr(
                f"Calculation error, not enough joint angles to build control vector: {e}"
            )
            return None
        except Exception as e:
            rospy.logerr(f"An error occurred during calculation: {e}")
            return None

    def handle_interpret_request(self, req):
        """
        服务处理函数。现在它只负责解析请求，调用核心计算函数，并打包响应。
        """
        rospy.loginfo("Handling service request.")
        response = Topo2AnglesResponse()
        try:
            num_cols = req.graph_data.edges.layout.dim[1].size
            edges = np.array(req.graph_data.edges.data, dtype=np.double).reshape(
                -1, num_cols
            )
            leg_angel = np.array(req.graph_data.feature.leg_angles, dtype=np.double)

            # 调用核心计算函数
            control_vector = self._calculate_control_vector(edges, leg_angel)

            if control_vector is not None:
                response.angles = np.array(control_vector, dtype=np.float32).tolist()
                rospy.loginfo("Service request handled successfully.")
            else:
                response.angles = []
        except Exception as e:
            rospy.logerr(f"Error parsing service request: {e}")
            response.angles = []

        return response

    def optimal_topo_callback(self, msg: TopologicalGraph):
        """
        这是订阅到 /optimal_topology 话题后的回调函数。
        """
        rospy.loginfo("Received message on /optimal_topology topic.")
        try:
            num_cols = msg.edges.layout.dim[1].size
            edges = np.array(msg.edges.data, dtype=np.double).reshape(-1, num_cols)
            leg_angel = np.array(msg.feature.leg_angles, dtype=np.double)

            # 调用同一个核心计算函数
            control_vector = self._calculate_control_vector(edges, leg_angel)

            # 如果计算成功，就发布 motor_ctrl 指令
            if control_vector is not None:
                motor_msg = Float32MultiArray()
                motor_msg.data = control_vector
                self.motor_ctrl_publisher.publish(motor_msg)
                rospy.loginfo(
                    f"Published motor_ctrl command with {len(control_vector)} elements."
                )

        except Exception as e:
            rospy.logerr(f"Error processing message from topic: {e}")


if __name__ == "__main__":
    try:
        KinematicsInterpreterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
