#!/usr/bin/env python3

from std_msgs.msg import Float32MultiArray
from meta_msgs.srv import TransPlanner, TransPlannerResponse
from dynamixel_msgs.srv import GetPos
from dynamixel_msgs.msg import SetParam
import rospy
import numpy as np

from interpolation import Interpolation
from utils.sim2real import sim2real, real2sim


class Planner:
    def __init__(self):
        self.flag_sim = rospy.get_param("flag_sim", True)
        self.step_time = rospy.get_param("step_time", 0.05)
        rospy.loginfo("TransPlanner initialized with flag_sim: %s", self.flag_sim)
        if self.flag_sim:
            from crimson_sim.srv import get_angles

            self.sim_pub = rospy.Publisher(
                "planned_sim_actuator", Float32MultiArray, queue_size=10
            )
            rospy.loginfo("Using simulated actuator publisher")
        else:
            self.real_pub = rospy.Publisher(
                "/dynamixel_control/sync_write", SetParam, queue_size=10
            )
            rospy.loginfo("Using real actuator publisher")

        self.optimal_angles_sub = rospy.Subscriber(
            "motor_ctrl", Float32MultiArray, self.optimal_callback
        )
        self.planner_service = rospy.Service(
            "trans_planner", TransPlanner, self.handle_planner_request
        )

    def optimal_callback(self, msg):
        # Handle the incoming optimal_angles message
        optimal_angles = msg.data
        rospy.loginfo("Received optimal angles: %s", optimal_angles)
        actuator_commands = self.planner(optimal_angles, steps=100)
        for command in actuator_commands:
            if self.flag_sim:
                self.sim_pub.publish(Float32MultiArray(data=command))
                rospy.loginfo("Published simulated actuator command: %s", command)
            else:
                set_param = SetParam()
                set_param.paramType = 1
                set_param.motorID = [i for i in range(1, 18)]
                set_param.params = sim2real(command)
                self.real_pub.publish(set_param)
                # rospy.loginfo("Published real actuator command: %s", command)
            rospy.sleep(self.step_time)

    def handle_planner_request(self, request):
        target = request.target
        steps = request.steps

        trajectory = self.planner(target, steps)
        response_trajectory = Float32MultiArray(data=trajectory)

        return TransPlannerResponse(trajectory=response_trajectory)

    def planner(self, target, steps):
        if self.flag_sim:
            # call get_angles service
            rospy.wait_for_service("get_angles")
            try:
                get_angles_service = rospy.ServiceProxy("get_angles", get_angles)
                response = get_angles_service()
                curr = response.angles
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s", e)
                return []
            # Create a temporary planner instance to use linear_interpolation
            temp_planner = Interpolation(self.flag_sim, curr, target, steps)
            actuator = temp_planner.gen_trajectory()
        else:
            # call GetPos service
            rospy.wait_for_service("/dynamixel_control/pos")
            try:
                get_pos_service = rospy.ServiceProxy("/dynamixel_control/pos", GetPos)
                response = get_pos_service()
                curr = real2sim(response.pos)
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s", e)
                return []
            # Create a temporary planner instance to use linear_interpolation
            temp_planner = Interpolation(self.flag_sim, curr, target, steps)
            actuator = temp_planner.gen_trajectory()

        print("Actuator commands:", actuator)
        return actuator


def main():
    rospy.init_node("trans_planner")
    Planner()
    rospy.loginfo("TransPlanner node started")
    rospy.spin()


if __name__ == "__main__":
    main()
