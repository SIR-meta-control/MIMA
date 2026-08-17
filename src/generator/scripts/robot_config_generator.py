#!/usr/bin/env python3
"""ROS adapter from requirement-vector topics to the GVAE HTTP service."""

from __future__ import annotations

import sys
from pathlib import Path

import rospy
from meta_msgs.msg import Global as GlobalFeature
from meta_msgs.msg import TopologicalGraph, TopoList
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout


def find_package_dir():
    source_candidate = Path(__file__).resolve().parents[1]
    if (source_candidate / "generation" / "service_client.py").is_file():
        return source_candidate
    try:
        import rospkg
    except ImportError as exc:
        raise RuntimeError("rospkg is required to locate the generator package") from exc
    try:
        return Path(rospkg.RosPack().get_path("generator"))
    except rospkg.ResourceNotFound as exc:
        raise RuntimeError("could not locate the generator ROS package") from exc


GENERATION_DIR = find_package_dir() / "generation"
if str(GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATION_DIR))

from service_client import GeneratorServiceClient, GeneratorServiceError
from service_contract import mima_vector_to_vreq, validate_vreq


ADJACENCY = {
    "4-bar": [2, 3, 3, 4, 4, 5],
    "8-bar": [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 1],
    "6-bar": [2, 3, 3, 4, 4, 5, 5, 7, 7, 8, 8, 2],
}


def requirement_vector_to_vreq(values):
    """Accept native GVAE-6 or MIMA-7; MIMA ``hs`` is intentionally ignored."""
    values = list(values)
    if len(values) == 6:
        return validate_vreq(values)
    if len(values) == 7:
        return mima_vector_to_vreq(values)
    raise ValueError(
        "requirement vector must be GVAE-6 or MIMA-7, got %d values" % len(values)
    )


def matrix_message(values, row_label, expected_rows, expected_columns):
    if not isinstance(values, list) or len(values) != expected_rows:
        raise ValueError("%s must contain %d rows" % (row_label, expected_rows))
    rows = []
    for index, row in enumerate(values):
        if not isinstance(row, list) or len(row) != expected_columns:
            raise ValueError(
                "%s[%d] must contain %d values"
                % (row_label, index, expected_columns)
            )
        rows.append([float(value) for value in row])

    message = Float32MultiArray()
    message.layout = MultiArrayLayout()
    message.layout.dim = [
        MultiArrayDimension(
            label=row_label,
            size=expected_rows,
            stride=expected_rows * expected_columns,
        ),
        MultiArrayDimension(
            label="features",
            size=expected_columns,
            stride=expected_columns,
        ),
    ]
    message.layout.data_offset = 0
    message.data = [value for row in rows for value in row]
    return message


def candidate_to_topology(candidate):
    structure = candidate["structure"]
    global_data = structure["global"]
    edge_rows = structure["edges"]
    if len(edge_rows) != 8:
        raise ValueError("candidate must contain 8 edges")

    # The service retains [angle, xyz, quaternion]. Existing MIMA downstream
    # nodes run with flag_test=false and consume [xyz, quaternion].
    edge_poses = []
    for index, row in enumerate(edge_rows):
        if len(row) == 8:
            edge_poses.append(row[1:])
        elif len(row) == 7:
            edge_poses.append(row)
        else:
            raise ValueError("edge[%d] must contain 7 or 8 values" % index)

    topology = TopologicalGraph()
    topology.nodes = matrix_message(structure["nodes"], "nodes", 8, 7)
    topology.edges = matrix_message(edge_poses, "edges", 8, 7)
    topology.adjacency = Float32MultiArray()
    topology.adjacency.data = [
        float(value) for value in ADJACENCY[candidate["bar_type"]]
    ]

    scale = list(global_data["scale"])
    leg_angles = list(global_data["leg_angle"])
    if len(scale) != 3 or len(leg_angles) != 3:
        raise ValueError("scale and leg_angle must each contain 3 values")
    feature = GlobalFeature()
    feature.scale = [float(value) for value in scale]
    feature.leg_angles = [float(value) for value in leg_angles]
    feature.leg_base = matrix_message(global_data["leg_base"], "legs", 4, 7)
    feature.locomotion_mode = 0
    topology.feature = feature
    return topology


class RobotConfigGenerator:
    def __init__(self):
        self.service_url = rospy.get_param(
            "~service_url", "http://127.0.0.1:8091"
        )
        self.service_timeout_s = float(
            rospy.get_param("~service_timeout_s", 120.0)
        )
        self.samples_per_bar = int(rospy.get_param("~samples_per_bar", 64))
        self.top_k = int(rospy.get_param("~top_k", 10))
        self.bar_types = rospy.get_param("~bar_types", "auto")
        self.temperature = float(rospy.get_param("~temperature", 1.0))
        self.diversity_threshold = float(
            rospy.get_param("~diversity_threshold", 0.02)
        )
        self.min_per_bar = int(rospy.get_param("~min_per_bar", 1))
        self.seed = int(rospy.get_param("~seed", 7))
        self.input_topic = rospy.get_param(
            "~requirement_vector_topic", "/requirement/vector"
        )
        self.output_topic = rospy.get_param(
            "~generated_configs_topic", "/generated_topolist"
        )

        self.client = GeneratorServiceClient(
            self.service_url,
            timeout_s=self.service_timeout_s,
        )
        self.config_pub = rospy.Publisher(
            self.output_topic,
            TopoList,
            queue_size=10,
        )
        self.vector_sub = rospy.Subscriber(
            self.input_topic,
            Float32MultiArray,
            self.vector_callback,
            queue_size=1,
        )

        try:
            health = self.client.health()
            rospy.loginfo(
                "GVAE service connected: model=%s device=%s",
                health.get("model", "unknown"),
                health.get("device", "unknown"),
            )
        except GeneratorServiceError as exc:
            rospy.logwarn(
                "GVAE service is not ready at startup (%s); callbacks will retry",
                exc,
            )
        rospy.loginfo(
            "Generator adapter ready: %s -> %s via %s",
            self.input_topic,
            self.output_topic,
            self.service_url,
        )

    def vector_callback(self, message):
        try:
            vreq = requirement_vector_to_vreq(message.data)
            response = self.client.generate(
                vreq,
                samples_per_bar=self.samples_per_bar,
                top_k=self.top_k,
                bar_types=self.bar_types,
                temperature=self.temperature,
                diversity_threshold=self.diversity_threshold,
                min_per_bar=self.min_per_bar,
                seed=self.seed,
            )
            topology_list = TopoList()
            topology_list.graphs = [
                candidate_to_topology(candidate)
                for candidate in response["candidates"]
            ]
            if not topology_list.graphs:
                rospy.logwarn(
                    "GVAE returned no valid candidates: %s",
                    response.get("summary", {}).get("rejection_counts", {}),
                )
                return
            self.config_pub.publish(topology_list)
            summary = response["summary"]
            rospy.loginfo(
                "Published %d/%d valid GVAE candidates (generated=%d)",
                len(topology_list.graphs),
                summary.get("valid", 0),
                summary.get("generated", 0),
            )
        except (ValueError, KeyError, TypeError, GeneratorServiceError) as exc:
            rospy.logerr("Generator request failed: %s", exc)
        except Exception as exc:
            rospy.logerr("Unexpected generator adapter failure: %s", exc)

    @staticmethod
    def run():
        rospy.spin()


def main():
    rospy.init_node("robot_config_generator")
    RobotConfigGenerator().run()


if __name__ == "__main__":
    main()
