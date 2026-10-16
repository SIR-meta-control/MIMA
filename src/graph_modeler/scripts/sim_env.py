import numpy as np
import mujoco
from pathlib import Path
from gymnasium.envs.mujoco import MujocoEnv
from scipy.spatial.transform import Rotation
import sys

_ROS_MUJOCO_SCRIPTS = Path(__file__).resolve().parents[2] / "ros_mujoco/scripts"
if _ROS_MUJOCO_SCRIPTS.is_dir():
    sys.path.insert(0, str(_ROS_MUJOCO_SCRIPTS))

from ros_mujoco_utils.model_assets import find_crimson_scene

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
    """Crimson capture accessors backed by Gymnasium's MuJoCo runtime.

    Capture ordering follows the original crimson_sim environment. Joint
    poses refer to their owning bodies; node and leg-base poses refer to
    named sites. All poses are in world coordinates with wxyz quaternions.
    """

    FRAME_JOINTS = tuple(f"frameJoint{i}" for i in range(2, 9))
    NODE_SITES = tuple(f"NodeSite{i}" for i in range(1, 9))
    LEG_BASE_SITES = (
        "DimSite_leg_16",
        "DimSite_leg_15",
        "DimSite_leg_17",
        "DimSite_leg_14",
    )

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, **kwargs):
        model_path = find_crimson_scene()
        MujocoEnv.__init__(
            self,
            model_path=model_path.absolute().as_posix(),
            frame_skip=10,  # Physics steps per evaluation frame; timestep is set by the MJCF.
            observation_space=None,  # Manually set afterwards
            # default_camera_config=DEFAULT_CAMERA_CONFIG,
            default_camera_config=None,
            **kwargs,
        )

        # Resolve every required element up front instead of silently dropping
        # missing rows or reading the last element through an ID of -1.
        self._frame_joint_ids = self._named_ids(
            mujoco.mjtObj.mjOBJ_JOINT, self.FRAME_JOINTS
        )
        self._node_site_ids = self._named_ids(mujoco.mjtObj.mjOBJ_SITE, self.NODE_SITES)
        self._leg_base_site_ids = self._named_ids(
            mujoco.mjtObj.mjOBJ_SITE, self.LEG_BASE_SITES
        )
        mujoco.mj_forward(self.model, self.data)

    def _named_ids(self, object_type, names):
        ids = np.array(
            [mujoco.mj_name2id(self.model, object_type, name) for name in names],
            dtype=int,
        )
        missing = [name for name, identifier in zip(names, ids) if identifier < 0]
        if missing:
            self.close()
            raise ValueError("Crimson capture model is missing: " + ", ".join(missing))
        return ids

    @staticmethod
    def _world_quaternions(matrices):
        quaternions = np.empty((len(matrices), 4))
        for quaternion, matrix in zip(quaternions, matrices):
            mujoco.mju_mat2Quat(quaternion, matrix)
        return quaternions

    def get_joint_pos(self):
        body_ids = self.model.jnt_bodyid[self._frame_joint_ids]
        return self.data.xpos[body_ids].copy()

    def get_joint_quat(self):
        body_ids = self.model.jnt_bodyid[self._frame_joint_ids]
        return self._world_quaternions(self.data.xmat[body_ids])

    def get_joint_angles(self):
        # Joint IDs are not qpos addresses when a free root precedes hinges.
        return self.data.qpos[self.model.jnt_qposadr[self._frame_joint_ids]].copy()

    def get_node_pos(self):
        return self.data.site_xpos[self._node_site_ids].copy()

    def get_node_quat(self):
        return self._world_quaternions(self.data.site_xmat[self._node_site_ids])

    def get_simple_size_from_site(self):
        """World-axis extents of all site centers, as in the original capture."""
        return np.ptp(self.data.site_xpos, axis=0)

    def get_leg_base_site_pos(self):
        return self.data.site_xpos[self._leg_base_site_ids].copy()

    def get_leg_base_site_quat(self):
        # Retain the original SciPy quaternion sign convention for this field.
        matrices = self.data.site_xmat[self._leg_base_site_ids].reshape(-1, 3, 3)
        xyzw = Rotation.from_matrix(matrices).as_quat()
        return xyzw[:, [3, 0, 1, 2]]

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
