import numpy as np
import mujoco
from pathlib import Path
from gymnasium.envs.mujoco import MujocoEnv
from scipy.optimize import least_squares
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
    """Custom Environment that follows gym interface."""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, control_fps=None, **kwargs):
        if control_fps is not None:
            if (
                isinstance(control_fps, bool)
                or not np.isfinite(control_fps)
                or control_fps <= 0
            ):
                raise ValueError("control_fps must be finite and positive.")
        # 获取当前脚本的绝对路径，向上推三层到达 src 目录，再进入 models
        model_path = find_crimson_scene()

        MujocoEnv.__init__(
            self,
            model_path=model_path.absolute().as_posix(),
            frame_skip=10,  # Retain the standalone replay default unless a control rate is supplied.
            observation_space=None,  # Manually set afterwards
            # default_camera_config=DEFAULT_CAMERA_CONFIG,
            default_camera_config=None,
            **kwargs,
        )

        if control_fps is not None:
            substeps = 1.0 / (float(control_fps) * self.model.opt.timestep)
            frame_skip = int(round(substeps))
            if frame_skip < 1 or not np.isclose(
                substeps, frame_skip, rtol=0.0, atol=1e-9
            ):
                self.close()
                raise ValueError(
                    "Control period must be an integer number of physics steps."
                )
            self.frame_skip = frame_skip

    def set_stopped_pose(self, control, closure_tolerance_m=1e-3):
        """Set measured actuated positions, solve passive joints, and start at rest."""
        control = np.asarray(control, dtype=float).reshape(-1)
        if control.shape != (17,) or not np.isfinite(control).all():
            raise ValueError("Initial pose must contain 17 finite joint angles.")
        if self.model.nu != 17:
            raise ValueError("The Crimson model must have 17 actuators.")
        if not np.isfinite(closure_tolerance_m) or closure_tolerance_m <= 0:
            raise ValueError("closure_tolerance_m must be finite and positive.")
        if not np.all(self.model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT):
            raise ValueError("Initial pose requires joint-transmission actuators.")

        joint_ids = self.model.actuator_trnid[:, 0]
        if not np.all(self.model.jnt_type[joint_ids] == mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Initial pose requires hinge-joint actuators.")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.model.jnt_qposadr[joint_ids]] = control
        self.data.ctrl[:] = control
        passive_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ("frameJoint4", "frameJoint7")
        ]
        if any(joint < 0 for joint in passive_ids):
            raise ValueError("The Crimson passive frame joints are missing.")
        passive_addresses = self.model.jnt_qposadr[passive_ids]

        def closure_residual(angles):
            self.data.qpos[passive_addresses] = angles
            mujoco.mj_forward(self.model, self.data)
            mask = self.data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
            return self.data.efc_pos[mask].copy()

        # The nominal closed-chain relation supplies a nearby branch for fitting.
        initial = np.array([-control[1] - control[2], control[2]])
        fit = least_squares(
            closure_residual,
            initial,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=150,
        )
        residual = closure_residual(fit.x)
        self.initial_closure_error_m = float(np.max(np.abs(residual)))
        if not fit.success or self.initial_closure_error_m > closure_tolerance_m:
            raise ValueError(
                "Current joint pose cannot satisfy the model loop closure: "
                "maximum residual %.6f m (limit %.6f m)."
                % (self.initial_closure_error_m, closure_tolerance_m)
            )
        self._place_on_ground()
        self.data.qvel[:] = 0.0
        self.data.qacc_warmstart[:] = 0.0
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _place_on_ground(self):
        """Place the nominally oriented floating base on the scene's flat floor."""
        floor_ids = np.flatnonzero(
            (self.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
            & (self.model.geom_bodyid == 0)
        )
        if len(floor_ids) != 1:
            raise ValueError(
                "Stationary initialization requires one horizontal ground plane."
            )
        floor = int(floor_ids[0])
        rotation = self.data.geom_xmat[floor].reshape(3, 3)
        if not np.allclose(rotation[:, 2], [0, 0, 1], rtol=0.0, atol=1e-9):
            raise ValueError(
                "Stationary initialization requires a horizontal ground plane."
            )
        lowest = float("inf")
        for geom in range(self.model.ngeom):
            if self.model.geom_bodyid[geom] == 0:
                continue
            if not (self.model.geom_contype[geom] or self.model.geom_conaffinity[geom]):
                continue
            rotation = self.data.geom_xmat[geom].reshape(3, 3)
            if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = self.model.geom_dataid[geom]
                start = self.model.mesh_vertadr[mesh]
                vertices = self.model.mesh_vert[
                    start : start + self.model.mesh_vertnum[mesh]
                ]
                min_z = (vertices @ rotation[2]).min() + self.data.geom_xpos[geom, 2]
            else:
                center, extent = np.split(self.model.geom_aabb[geom], 2)
                min_z = (
                    self.data.geom_xpos[geom, 2]
                    + rotation[2] @ center
                    - np.abs(rotation[2]) @ extent
                )
            lowest = min(lowest, float(min_z))
        root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        if root < 0 or self.model.jnt_type[root] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("The Crimson floating root joint is missing.")
        if not np.isfinite(lowest):
            raise ValueError(
                "No robot collision geometry is available for ground placement."
            )
        root_z = self.model.jnt_qposadr[root] + 2
        self.data.qpos[root_z] += self.data.geom_xpos[floor, 2] - lowest
        mujoco.mj_forward(self.model, self.data)

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
