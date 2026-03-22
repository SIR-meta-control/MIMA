import numpy as np
import mujoco
from pathlib import Path
from gymnasium.envs.mujoco import MujocoEnv
from scipy.spatial.transform import Rotation as R

DEFAULT_CAMERA_CONFIG = {
    "azimuth": 90.0,
    "distance": 3.0,
    "elevation": -25.0,
    "lookat": np.array([0.0, 0.0, 0.0]),
    "fixedcamid": 0,
    "trackbodyid": -1,
    "type": 2,
}

file_path = "recordings/pointsInfo/velocity_data.txt"


class CrimsonMujocoEnv(MujocoEnv):
    """Custom Environment that follows gym interface."""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, **kwargs):
        # 获取当前脚本的绝对路径，向上推三层到达 src 目录，再进入 models
        base_dir = Path(__file__).resolve().parent.parent.parent
        model_path = base_dir / "models" / "crimson" / "mjcf" / "crimson_scene.xml"

        MujocoEnv.__init__(
            self,
            model_path=model_path.absolute().as_posix(),
            frame_skip=10,  # Perform an action every 10 frames (dt(=0.002) * 10 = 0.02 seconds -> 50hz action rate)
            observation_space=None,  # Manually set afterwards
            # default_camera_config=DEFAULT_CAMERA_CONFIG,
            default_camera_config=None,
            **kwargs,
        )

    def get_joint_torque(self):
        """获取指定关节对应执行器所施加的扭矩"""
        # 关节名称列表
        target_joints = [
            "frameJoint2",
            "frameJoint3",
            "frameJoint5",
            "frameJoint6",
            "frameJoint8",
            "leg0Joint1",
            "leg0Joint2",
            "leg0Joint3",
            "leg1Joint1",
            "leg1Joint2",
            "leg1Joint3",
            "leg2Joint1",
            "leg2Joint2",
            "leg2Joint3",
            "leg3Joint1",
            "leg3Joint2",
            "leg3Joint3",
        ]
        # 存储扭矩的列表
        joint_torques = []

        for joint_name in target_joints:
            # 1. 获取关节ID
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )

            if joint_id == -1:
                print(f"Warning: Joint '{joint_name}' not found!")
                joint_torques.append(0.0)
                continue

            # 2. 遍历所有执行器，寻找控制当前关节的那一个
            #    这个内部循环对应您代码中的 body_id = self.model.jnt_bodyid[joint_id]
            #    因为没有从joint_id到act_id的直接映射，所以需要搜索
            found_actuator = False
            for act_id in range(self.model.nu):  # self.model.nu 是执行器的总数
                # 检查这个执行器(act_id)的目标是不是我们正在找的关节(joint_id)
                is_joint_actuator = (
                    self.model.actuator_trntype[act_id] == mujoco.mjtTrn.mjTRN_JOINT
                )
                is_correct_joint = self.model.actuator_trnid[act_id, 0] == joint_id

                if is_joint_actuator and is_correct_joint:
                    # 3. 从data.actuator_force中获取扭矩
                    torque = self.data.actuator_force[act_id]
                    joint_torques.append(torque)
                    found_actuator = True
                    break  # 跳出内部的执行器循环，开始找下一个关节

            # 4. 如果遍历完所有执行器都没找到匹配的
            if not found_actuator:
                print(f"Warning: No actuator found for joint '{joint_name}'.")
                joint_torques.append(0.0)

        # 5. 转换为numpy数组并返回
        return np.array(joint_torques)


if __name__ == "__main__":
    env = CrimsonMujocoEnv(render_mode="human")
    for t in range(int(1e9)):
        env.get_leg_base_site_pos()
    env.close()
