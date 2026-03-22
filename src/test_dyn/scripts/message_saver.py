#!/usr/bin/env python3
import rospy
import json
import os
import datetime
from dynamixel_msgs.msg import SetParam
from meta_msgs.msg import TopoList
from std_msgs.msg import Float32MultiArray
from meta_msgs.msg import TopologicalGraph


class MessageSaver:
    def __init__(self):
        rospy.init_node("message_saver_node", anonymous=True)

        # Define topics
        self.sync_write_topic = "/dynamixel_control/sync_write"
        self.generated_topolist_topic = "/generated_topolist"
        self.motor_ctrl_topic = "/motor_ctrl"
        self.optimal_topology_topic = "/optimal_topology"

        # Initialize storage
        self.data = {
            "sync_write": [],
            "generated_topolist": [],
            "motor_ctrl": [],
            "optimal_topology": [],
        }

        # Setup output file
        workspace_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../..")
        )
        runs_dir = os.path.join(workspace_root, "runs")
        if not os.path.exists(runs_dir):
            os.makedirs(runs_dir)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = os.path.join(runs_dir, f"result_{timestamp}.json")
        rospy.loginfo(f"Saving data to {self.output_file}")

        # Setup subscribers
        rospy.Subscriber(self.sync_write_topic, SetParam, self.sync_write_cb)
        rospy.Subscriber(
            self.generated_topolist_topic, TopoList, self.generated_topolist_cb
        )
        rospy.Subscriber(self.motor_ctrl_topic, Float32MultiArray, self.motor_ctrl_cb)
        rospy.Subscriber(
            self.optimal_topology_topic, TopologicalGraph, self.optimal_topology_cb
        )

        # Setup save timer (save periodically)
        self.save_timer = rospy.Timer(rospy.Duration(1.0), self.save_data)

        self.last_msg_time = rospy.get_time()
        self.timeout_timer = rospy.Timer(rospy.Duration(1.0), self.check_timeout)

    def check_timeout(self, event=None):
        if rospy.get_time() - self.last_msg_time > 10.0:
            rospy.loginfo(f"Data saved to {self.output_file}. Ctrl C to exit.")
            rospy.signal_shutdown("Timeout reached")

    def sync_write_cb(self, msg):
        self.last_msg_time = rospy.get_time()
        self.data["sync_write"].append(
            {
                "time": rospy.get_time(),
                # Extract relevant fields from SyncWrite msg, assuming typical structure. Update as needed.
                "data": str(msg),
            }
        )

    def generated_topolist_cb(self, msg):
        self.last_msg_time = rospy.get_time()
        self.data["generated_topolist"].append(
            {
                "time": rospy.get_time(),
                # Extract relevant fields. Update as needed.
                "data": str(msg),
            }
        )

    def motor_ctrl_cb(self, msg):
        self.last_msg_time = rospy.get_time()
        self.data["motor_ctrl"].append({"time": rospy.get_time(), "data": msg.data})

    def optimal_topology_cb(self, msg):
        self.last_msg_time = rospy.get_time()
        self.data["optimal_topology"].append(
            {
                "time": rospy.get_time(),
                # Extract relevant fields. Update as needed.
                "data": str(msg),
            }
        )

    def save_data(self, event=None):
        try:
            with open(self.output_file, "w") as f:
                json.dump(self.data, f, indent=4)
        except Exception as e:
            rospy.logerr(f"Failed to save data: {e}")

    def run(self):
        rospy.spin()
        # Final save on exit
        self.save_data()


if __name__ == "__main__":
    try:
        saver = MessageSaver()
        saver.run()
    except rospy.ROSInterruptException:
        pass
