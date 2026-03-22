#!/usr/bin/env python3

import rospy
import json
import numpy as np
from pathlib import Path

# 导入需要用到的消息类型
from meta_msgs.msg import TopologicalGraph, Global
from std_msgs.msg import MultiArrayDimension


def publish_optimal_topo():
    """
    读取JSON文件，打包成TopologicalGraph消息，并发布一次。
    """
    rospy.init_node("optimal_topo_publisher_test")

    # 创建到 /optimal_topology 话题的发布器
    pub = rospy.Publisher("/optimal_topology", TopologicalGraph, queue_size=10)

    # 等待有节点订阅这个话题
    rospy.sleep(1)

    # --- 加载并打包请求数据 (与之前的客户端代码一致) ---
    try:
        base_path = Path(__file__).parent
        json_path = base_path / "4bar.json"
        with open(json_path, "r") as f:
            graph_data = json.load(f)
    except Exception as e:
        rospy.logerr(f"Failed to load JSON file: {e}")
        return

    graph_msg = TopologicalGraph()
    edges_list = graph_data.get("edges", [])
    if edges_list:
        edges_np = np.array(edges_list, dtype=np.float32)
        graph_msg.edges.layout.dim.append(
            MultiArrayDimension(
                label="edges", size=edges_np.shape[0], stride=edges_np.size
            )
        )
        graph_msg.edges.layout.dim.append(
            MultiArrayDimension(
                label="edge_data", size=edges_np.shape[1], stride=edges_np.shape[1]
            )
        )
        graph_msg.edges.data = edges_np.flatten().tolist()
    global_data = graph_data.get("global", {})
    if "leg_angle" in global_data:
        graph_msg.feature.leg_angles = global_data["leg_angle"]

    # --- 发布消息 ---
    if pub.get_num_connections() > 0:
        rospy.loginfo("Publishing a test message to /optimal_topology...")
        pub.publish(graph_msg)
        rospy.loginfo("Message published.")
    else:
        rospy.logwarn("No subscribers found on /optimal_topology topic.")


if __name__ == "__main__":
    try:
        publish_optimal_topo()
    except rospy.ROSInterruptException:
        pass
