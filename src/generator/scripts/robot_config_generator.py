#!/home/inron/software/anaconda3/envs/mira/bin/python
import sys
import os
import rospy
import torch
import numpy as np
import glob
import json
from std_msgs.msg import Float32MultiArray
from meta_msgs.msg import Global as GlobalFeature
from meta_msgs.msg import TopologicalGraph, TopoList

current_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.dirname(current_dir)


generation_dir = os.path.join(package_dir, "generation")
if generation_dir not in sys.path:
    sys.path.insert(0, generation_dir)


class RobotConfigGenerator:
    def __init__(self):
        self.model_path = rospy.get_param("~model_path", "")
        self.num_configs = rospy.get_param("~num_configs", 10)

        self.input_topic = rospy.get_param(
            "~detection_vector_topic", "/detection/vector"
        )
        self.output_topic = rospy.get_param(
            "~generated_configs_topic", "/generated_topolist"
        )

        if not self.model_path:
            rospy.logerr("Robot Config Generator - model_path parameter is required!")
            return

        if not os.path.exists(self.model_path):
            rospy.logerr(
                f"Robot Config Generator - model file not found: {self.model_path}"
            )
            return

        try:
            from model import RobotConfigurationNet

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            checkpoint = torch.load(self.model_path, map_location="cpu")
            if "args" in checkpoint:
                model_args_dict = checkpoint["args"]
            else:
                rospy.logwarn(
                    "Robot Config Generator - no args found in checkpoint, using default values"
                )
                model_args_dict = {
                    "batch_size": 32,
                    "graph_imputation_path": "graph_imputation.npy",
                }

            if "graph_imputation_path" in model_args_dict:
                graph_path = model_args_dict["graph_imputation_path"]
                if not os.path.isabs(graph_path):
                    model_args_dict["graph_imputation_path"] = os.path.join(
                        generation_dir, graph_path
                    )

            class Args:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)

            model_args = Args(**model_args_dict)

            self.model = RobotConfigurationNet(
                model_args, num_configs=self.num_configs
            ).to(self.device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            self.vector_sub = rospy.Subscriber(
                self.input_topic,
                Float32MultiArray,
                self.vector_callback,
                queue_size=1,
            )

            self.config_pub = rospy.Publisher(
                self.output_topic, TopoList, queue_size=10
            )

            rospy.loginfo("Robot Config Generator - initialized successfully!")

        except Exception as e:
            rospy.logerr(f"Robot Config Generator - initialization failed: {e}")
            import traceback

            rospy.logerr(f"Traceback: {traceback.format_exc()}")
            return

    def vector_callback(self, msg):
        try:
            vreq = list(msg.data)

            configs = self.generate_configs(vreq)

            if configs:
                self.process_and_output_configs(configs, vreq)
            else:
                rospy.logerr(
                    "Robot Config Generator - failed to generate configurations"
                )

        except Exception as e:
            rospy.logerr(f"Robot Config Generator - vector processing error: {e}")
            import traceback

            rospy.logerr(f"Traceback: {traceback.format_exc()}")

    def generate_configs(self, vreq):
        try:
            with torch.no_grad():
                vreq_tensor = torch.tensor([vreq], dtype=torch.float32).to(self.device)

                all_configs, all_confidences = self.model(vreq_tensor, train=False)

                return [
                    (config, confidence)
                    for config, confidence in zip(all_configs, all_confidences)
                ]

        except Exception as e:
            rospy.logerr(f"Robot Config Generator - config generation error: {e}")
            import traceback

            rospy.logerr(f"Traceback: {traceback.format_exc()}")
            return None

    def save_config_to_msg(self, config):
        try:
            from std_msgs.msg import (
                Float32MultiArray,
                MultiArrayDimension,
                MultiArrayLayout,
            )

            nodes = config["nodes"].detach().cpu().numpy()[0]  # [8, 7]
            edges = config["edges"].detach().cpu().numpy()[0]  # [8, 7]
            leg_base = config["leg_base"].detach().cpu().numpy()[0]  # [4, 7]
            leg_angle = config["leg_angle"].detach().cpu().numpy()[0]  # [3]
            scale = config["scale"].detach().cpu().numpy()[0]  # [3]
            bar_type = config["bar"][0] if "bar" in config else "unknown"

            if bar_type == "4-bar":
                adjencency = [2, 3, 3, 4, 4, 5]
            elif bar_type == "8-bar":
                adjencency = [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 1]
            elif bar_type == "6-bar":
                adjencency = [2, 3, 3, 4, 4, 5, 5, 7, 7, 8, 8, 2]
            else:
                adjencency = []

            topo_msg = TopologicalGraph()

            topo_msg.nodes = Float32MultiArray()
            topo_msg.nodes.layout = MultiArrayLayout()
            topo_msg.nodes.layout.dim = [
                MultiArrayDimension(label="nodes", size=8, stride=56),
                MultiArrayDimension(label="features", size=7, stride=7),
            ]
            topo_msg.nodes.layout.data_offset = 0
            topo_msg.nodes.data = nodes.flatten().tolist()

            topo_msg.edges = Float32MultiArray()
            topo_msg.edges.layout = MultiArrayLayout()
            topo_msg.edges.layout.dim = [
                MultiArrayDimension(label="edges", size=8, stride=56),
                MultiArrayDimension(label="features", size=7, stride=7),
            ]
            topo_msg.edges.layout.data_offset = 0
            topo_msg.edges.data = edges.flatten().tolist()

            topo_msg.adjacency = Float32MultiArray()
            topo_msg.adjacency.data = [float(x) for x in adjencency]

            global_feature = GlobalFeature()
            global_feature.scale = scale.tolist()
            global_feature.leg_angles = leg_angle.tolist()
            global_feature.locomotion_mode = 0

            global_feature.leg_base = Float32MultiArray()
            global_feature.leg_base.layout = MultiArrayLayout()
            global_feature.leg_base.layout.dim = [
                MultiArrayDimension(label="legs", size=4, stride=28),
                MultiArrayDimension(label="features", size=7, stride=7),
            ]
            global_feature.leg_base.layout.data_offset = 0
            global_feature.leg_base.data = leg_base.flatten().tolist()

            topo_msg.feature = global_feature

            return topo_msg

        except Exception as e:
            rospy.logerr(f"Robot Config Generator - config conversion error: {e}")
            import traceback

            rospy.logerr(f"Traceback: {traceback.format_exc()}")
            return None

    def process_and_output_configs(self, configs, vreq):
        try:
            topo_list = TopoList()
            topo_list.graphs = []

            for i, (config, confidence) in enumerate(configs):
                try:
                    config_msg = self.save_config_to_msg(config)

                    if config_msg:
                        topo_list.graphs.append(config_msg)

                    else:
                        rospy.logerr(
                            f"  Failed to convert configuration {i + 1} to message"
                        )

                except Exception as e:
                    rospy.logerr(
                        f"Robot Config Generator - error processing config {i + 1}: {e}"
                    )
                    import traceback

                    rospy.logerr(f"Traceback: {traceback.format_exc()}")
                    continue

            if topo_list.graphs:
                self.config_pub.publish(topo_list)
            else:
                rospy.logerr("No valid configurations to publish")

        except Exception as e:
            rospy.logerr(f"Robot Config Generator - output processing error: {e}")
            import traceback

            rospy.logerr(f"Traceback: {traceback.format_exc()}")

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("robot_config_generator")

    generator = RobotConfigGenerator()
    if generator:
        generator.run()
