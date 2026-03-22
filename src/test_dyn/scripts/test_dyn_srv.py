#!/usr/bin/env python3
from dynamixel_msgs.srv import GetPos, GetPosResponse
import rospy

def handle_get_pos(req):
  rospy.loginfo("Received GetPos request")
  test_positions = [1024, 1024, 1536, 1024, 1536, 2560, 3123, 4000, 1536, 3123, 4000, 1536, 3123, 4000, 2560, 3123, 4000]
  return GetPosResponse(test_positions)

def main():
    rospy.init_node('test_dyn_node')
    rospy.loginfo("Test Dynamixel service node started")
    
    # Create a service that listens for GetPos requests
    rospy.Service('/dynamixel_control/pos', GetPos, handle_get_pos)
    
    rospy.loginfo("Service 'get_pos' is ready to receive requests")
    
    # Keep the node running
    rospy.spin()

if __name__ == "__main__":
    main()
