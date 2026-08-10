#!/usr/bin/env python3
"""
ROS 与 MuJoCo 的桥接节点。

节点订阅实物 Dynamixel 的 /dynamixel_control/sync_write 指令，
把编码器目标位置转换为 Crimson MuJoCo 模型需要的 17 维控制量，
再写入 env.data.ctrl 推动仿真。
"""

from pathlib import Path

import mujoco
import numpy as np
import rospy
from dynamixel_msgs.msg import SetParam
from std_msgs.msg import Float32MultiArray

from ros_mujoco_utils.conversion import update_mujoco_control_from_real
from ros_mujoco.srv import get_angles, get_anglesResponse


class RosMujocoBridge:
    def __init__(self):
        default_model_path = (
            Path(__file__).resolve().parents[2]
            / "models"
            / "crimson"
            / "mjcf"
            / "crimson_scene.xml"
        )

        self.model_path = Path(rospy.get_param("~model_path", str(default_model_path)))
        self.render = bool(rospy.get_param("~render", True))
        self.update_rate = float(rospy.get_param("~update_rate", 100.0))
        self.frame_skip = int(rospy.get_param("~frame_skip", 10))
        self.real_control_topic = rospy.get_param(
            "~real_control_topic", "/dynamixel_control/sync_write"
        )
        self.direct_control_topic = rospy.get_param("~direct_control_topic", "")
        self.get_angles_service = rospy.get_param("~get_angles_service", "get_angles")

        self.model = mujoco.MjModel.from_xml_path(self.model_path.as_posix())
        self.data = mujoco.MjData(self.model)
        self.latest_control = None
        self.viewer = None

        if self.model.nu != 17:
            rospy.logwarn(
                "Expected a 17D Crimson actuator model, but MuJoCo model.nu=%d.",
                self.model.nu,
            )

        if self.render:
            self._try_start_viewer()

        rospy.Subscriber(self.real_control_topic, SetParam, self.real_control_callback)
        if self.direct_control_topic:
            rospy.Subscriber(
                self.direct_control_topic, Float32MultiArray, self.direct_control_callback
            )

        rospy.Service(self.get_angles_service, get_angles, self.handle_get_angles)

        rospy.loginfo("ros_mujoco loaded model: %s", self.model_path)
        rospy.loginfo("Subscribed real control topic: %s", self.real_control_topic)
        if self.direct_control_topic:
            rospy.loginfo("Subscribed direct MuJoCo control topic: %s", self.direct_control_topic)
        rospy.loginfo("Providing service: %s", self.get_angles_service)

    def _try_start_viewer(self):
        try:
            from mujoco import viewer

            self.viewer = viewer.launch_passive(self.model, self.data)
        except Exception as exc:
            self.viewer = None
            rospy.logwarn("Failed to start MuJoCo viewer, continuing headless: %s", exc)

    def real_control_callback(self, msg):
        if msg.paramType != 1:
            rospy.logwarn_throttle(
                1.0,
                "Ignoring SetParam with paramType=%s; only GoalPosition paramType=1 is supported.",
                msg.paramType,
            )
            return

        try:
            current = self.latest_control
            if current is None and msg.motorID:
                current = self.data.ctrl.copy()
            self.latest_control = update_mujoco_control_from_real(
                msg.params, current_control=current, motor_ids=msg.motorID
            )
        except Exception as exc:
            rospy.logwarn_throttle(1.0, "Failed to convert real command: %s", exc)

    def direct_control_callback(self, msg):
        control = np.array(msg.data, dtype=np.float64).reshape(-1)
        if control.shape[0] != self.model.nu:
            rospy.logwarn_throttle(
                1.0,
                "Ignoring direct control with length %d; expected %d.",
                control.shape[0],
                self.model.nu,
            )
            return
        self.latest_control = control

    def handle_get_angles(self, _req):
        qpos = [float(value) for value in self.data.qpos.copy()]
        return get_anglesResponse(angles=qpos[7:])

    def step(self):
        if self.latest_control is None:
            rospy.loginfo_once("Waiting for MuJoCo control commands...")
            return

        if len(self.latest_control) != self.model.nu:
            rospy.logwarn_throttle(
                1.0,
                "Control length %d does not match model.nu=%d.",
                len(self.latest_control),
                self.model.nu,
            )
            return

        self.data.ctrl[:] = self.latest_control
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        if self.viewer is not None:
            try:
                self.viewer.sync()
            except Exception as exc:
                rospy.logwarn_throttle(1.0, "MuJoCo viewer sync failed: %s", exc)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


def main():
    rospy.init_node("ros_mujoco")
    bridge = RosMujocoBridge()
    rate = rospy.Rate(bridge.update_rate)

    try:
        while not rospy.is_shutdown():
            bridge.step()
            rate.sleep()
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
