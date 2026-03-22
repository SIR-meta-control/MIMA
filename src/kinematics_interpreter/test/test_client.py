#!/usr/bin/env python3

import sys
import rospy
import json
import numpy as np
from pathlib import Path

# 导入消息和服务类型
from meta_msgs.msg import TopologicalGraph, Global
from std_msgs.msg import MultiArrayDimension
from kinematics_interpreter.srv import InterpretKinematics


def call_interpretation_service():
    """
    准备数据，调用服务，并打印结果。
    """
    rospy.loginfo("Waiting for 'interpret_kinematics' service...")
    # 阻塞程序直到服务可用
    rospy.wait_for_service('interpret_kinematics')
    rospy.loginfo("Service is available.")

    try:
        # 创建一个服务的句柄（代理）
        interpret_kinematics = rospy.ServiceProxy('interpret_kinematics', InterpretKinematics)

        # --- 1. 加载并打包请求数据 (与之前的 publisher 类似) ---
        # 为了简单，我们直接从之前的代码复制数据打包逻辑
        try:
            base_path = Path(__file__).parent
            json_path = base_path / 'test.json'
            with open(json_path, 'r') as f:
                graph_data = json.load(f)
        except Exception as e:
            rospy.logerr(f"Failed to load JSON file: {e}")
            return

        graph_msg = TopologicalGraph()
        # 打包 edges
        edges_list = graph_data.get('edges', [])
        if edges_list:
            edges_np = np.array(edges_list, dtype=np.float32)
            graph_msg.edges.layout.dim.append(
                MultiArrayDimension(label="edges", size=edges_np.shape[0], stride=edges_np.size))
            graph_msg.edges.layout.dim.append(
                MultiArrayDimension(label="edge_data", size=edges_np.shape[1], stride=edges_np.shape[1]))
            graph_msg.edges.data = edges_np.flatten().tolist()

        # 打包 feature
        global_data = graph_data.get('global', {})
        if 'leg_angle' in global_data:
            graph_msg.feature.leg_angle = global_data['leg_angle']

        # --- 2. 调用服务并传递请求 ---
        rospy.loginfo("Calling service with graph data...")
        # 调用服务，req.graph_data 将被填充为我们创建的 graph_msg
        response = interpret_kinematics(graph_data=graph_msg)

        # --- 3. 处理并打印响应 ---
        if response.angles:
            rospy.loginfo("Service Response Received:")
            rospy.loginfo(f"  Calculated Angles ({len(response.angles)} elements):")
            # 格式化打印，方便查看
            formatted_angles = [f"{angle:.4f}" for angle in response.angles]
            print("  " + ", ".join(formatted_angles))
        else:
            rospy.logwarn("Service returned an empty list of angles, indicating an error.")

    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
    except Exception as e:
        rospy.logerr(f"An error occurred in the client: {e}")


if __name__ == "__main__":
    call_interpretation_service()