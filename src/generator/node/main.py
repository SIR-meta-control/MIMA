#!/home/inron/software/anaconda3/envs/mira/bin/python
import sys
import os
import rospy


current_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.dirname(current_dir)
scripts_path = os.path.join(package_dir, "scripts")
sys.path.insert(0, scripts_path)

from robot_config_generator import RobotConfigGenerator


if __name__ == "__main__":
    rospy.init_node("generator_node", anonymous=True)

    generator = RobotConfigGenerator()

    generator.run()
