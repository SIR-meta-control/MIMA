#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32MultiArray


def publish_detection_vector():
    """send detection vector to /detection/vector/manually topic"""

    rospy.init_node("manual_detection_publisher", anonymous=True)

    pub = rospy.Publisher("/detection/vector/manually", Float32MultiArray, queue_size=1)

    rospy.loginfo("Waiting for publisher to connect...")
    rospy.sleep(1.0)

    msg = Float32MultiArray()
    msg.data = [0.8, 0.8, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0]

    pub.publish(msg)
    rospy.loginfo("Published detection vector: %s", msg.data)

    rospy.sleep(0.5)

    rospy.loginfo("Manual detection vector published successfully!")


if __name__ == "__main__":
    try:
        publish_detection_vector()
    except rospy.ROSInterruptException:
        rospy.logerr("Manual publisher interrupted")
    except Exception as e:
        rospy.logerr("Error in manual publisher: %s", e)
