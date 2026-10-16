#!/usr/bin/env python3
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Support catkin's relay and installed helpers beside the executable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import rospy

from meta_msgs.msg import TopoList, TopologicalGraph
from meta_msgs.srv import Topo2Angles
from dynamixel_msgs.srv import GetPos

# Support direct source-checkout execution as well as catkin's Python path.
_ROS_MUJOCO_SCRIPTS = Path(__file__).resolve().parents[2] / "ros_mujoco" / "scripts"
if _ROS_MUJOCO_SCRIPTS.is_dir():
    sys.path.insert(0, str(_ROS_MUJOCO_SCRIPTS))

from ros_mujoco_utils.energy_calibration import frame_energy_j
from sim_env import CrimsonMujocoEnv
from utils.sim2real import real2sim


OPTIMIZER_ENERGY_MODE = "dynamic_calibrated"


def constrain_judge(pos):
    """17D约束检查：phi_1=phi_2 且 phi+phi_1+phi_2=0。"""
    phi_1 = pos[1]
    phi_2 = pos[2]
    phi = pos[4]
    return np.isclose(phi_1, phi_2, atol=1e-03) and np.isclose(
        phi + phi_1 + phi_2, 0, atol=1e-03
    )


def _aligned_pose(pos):
    """Use the same axis-alignment waypoint as the execution planner."""
    aligned = pos.copy()
    aligned[1] = aligned[2] = (pos[1] + pos[2]) / 2.0
    aligned[3] = pos[0]
    aligned[4] = -(aligned[1] + aligned[2])
    return aligned


def torque_to_energy(torque, fps=50):
    """Return total robot energy per evaluation frame for a 17D torque vector.

    Use the same aggregate calibration as the system-level ablation. Arbitrary
    candidate trajectories have no transformation-loop interval label, so the
    loop-specific residual corrections are not applied.
    """
    return frame_energy_j(torque, fps=fps, mode=OPTIMIZER_ENERGY_MODE, transition=None)


def build_trajectory(q0_17, qt_17, transition_steps=400, constraint_steps=100):
    """Plan continuous start-alignment, main-motion, and final-motion segments."""
    q0 = np.asarray(q0_17, dtype=float).reshape(-1)
    qt = np.asarray(qt_17, dtype=float).reshape(-1)
    if q0.size != 17 or qt.size != 17:
        raise ValueError("build_trajectory requires 17D q0 and qt.")
    if not np.isfinite(q0).all() or not np.isfinite(qt).all():
        raise ValueError("Joint positions must be finite.")
    for name, steps in (
        ("transition_steps", transition_steps),
        ("constraint_steps", constraint_steps),
    ):
        if (
            isinstance(steps, bool)
            or not isinstance(steps, (int, np.integer))
            or steps < 2
        ):
            raise ValueError("%s must be an integer >= 2." % name)

    start = q0 if constrain_judge(q0) else _aligned_pose(q0)
    goal = qt if constrain_judge(qt) else _aligned_pose(qt)
    segments = []
    if not constrain_judge(q0):
        segments.append(np.linspace(q0, start, constraint_steps))
    segments.append(np.linspace(start, goal, transition_steps))
    if not constrain_judge(qt):
        segments.append(np.linspace(goal, qt, constraint_steps))
    return np.vstack(segments)


def compute_energy_for_trajectory(trajectory, sustain_steps=500, fps=50, render=False):
    """Evaluate motion and hold from the measured, stationary initial pose."""
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] != 17 or not len(trajectory):
        raise ValueError("trajectory must be a non-empty N x 17 array.")
    if not np.isfinite(trajectory).all():
        raise ValueError("trajectory must contain finite joint positions.")
    if (
        isinstance(sustain_steps, bool)
        or not isinstance(sustain_steps, (int, np.integer))
        or sustain_steps < 0
    ):
        raise ValueError("sustain_steps must be a non-negative integer.")
    if isinstance(fps, bool) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive.")

    env = CrimsonMujocoEnv(
        control_fps=float(fps), render_mode="human" if render else None
    )
    try:
        if not np.isclose(env.dt, 1.0 / fps, rtol=0.0, atol=1e-12):
            raise ValueError(
                "Simulation dt does not match the requested control period."
            )
        env.set_stopped_pose(trajectory[0])
        integration_fps = 1.0 / env.dt
        energy_trans = 0.0
        for pose in trajectory:
            env.data.ctrl[:] = pose
            env.do_simulation(env.data.ctrl, env.frame_skip)
            energy_trans += torque_to_energy(
                env.get_joint_torque(), fps=integration_fps
            )
            if render:
                env.render()

        energy_sustain = 0.0
        for _ in range(sustain_steps):
            env.data.ctrl[:] = trajectory[-1]
            env.do_simulation(env.data.ctrl, env.frame_skip)
            energy_sustain += torque_to_energy(
                env.get_joint_torque(), fps=integration_fps
            )
            if render:
                env.render()
        return float(energy_trans + energy_sustain)
    finally:
        env.close()


class TopologyProcessor:
    """单个拓扑处理任务：topo -> qt(17D) -> trajectory -> energy。"""

    def __init__(
        self,
        q0_17,
        topo,
        kinematic_service,
        transition_steps=400,
        constraint_steps=100,
        sustain_steps=500,
        fps=50,
        render=False,
        index=None,
    ):
        self.q0 = q0_17
        self.topo = topo
        self.kinematic_service = kinematic_service
        self.transition_steps = transition_steps
        self.constraint_steps = constraint_steps
        self.sustain_steps = sustain_steps
        self.fps = fps
        self.render = render
        self.index = index

        self.energy = None
        self.exception = None

    def process(self):
        rospy.loginfo("[Task %s] Start processing topology...", self.index)
        try:
            resp = self.kinematic_service(self.topo)
            qt = np.array(resp.angles, dtype=float).reshape(-1)

            if qt.shape[0] != 17:
                raise ValueError(
                    "interpret_kinematics must return 17 angles, got %d" % qt.shape[0]
                )

            trajectory = build_trajectory(
                self.q0,
                qt,
                transition_steps=self.transition_steps,
                constraint_steps=self.constraint_steps,
            )

            self.energy = compute_energy_for_trajectory(
                trajectory,
                sustain_steps=self.sustain_steps,
                fps=self.fps,
                render=self.render,
            )
            rospy.loginfo(
                "[Task %s] Energy calculated: %.4f J", self.index, self.energy
            )
            return self
        except Exception:
            self.exception = traceback.format_exc()
            rospy.logwarn(
                "[Task %s] Processing failed:\n%s", self.index, self.exception
            )
            return self


class ParetoTopologyOptimizerNode:
    def __init__(self):
        rospy.loginfo("Waiting for interpret_kinematics service...")
        rospy.wait_for_service("interpret_kinematics")
        self.kinematic_interpreter_srv = rospy.ServiceProxy(
            "interpret_kinematics", Topo2Angles
        )
        rospy.loginfo("interpret_kinematics service connected.")

        dxl_get_pos_service = "/dynamixel_control/pos"
        rospy.loginfo("Waiting for %s service...", dxl_get_pos_service)
        rospy.wait_for_service(dxl_get_pos_service)
        self.get_joints_srv = rospy.ServiceProxy(dxl_get_pos_service, GetPos)
        rospy.loginfo("get_pos service connected.")

        self.opt_pub = rospy.Publisher(
            "optimal_topology", TopologicalGraph, queue_size=10
        )
        self.sub = rospy.Subscriber(
            "generated_topolist", TopoList, self.topo_callback, queue_size=5
        )

        self.max_workers = rospy.get_param("~max_workers", 4)
        self.transition_steps = rospy.get_param("~transition_steps", 400)
        self.constraint_steps = rospy.get_param("~constraint_steps", 100)
        self.sustain_steps = rospy.get_param("~sustain_steps", 500)
        self.fps = rospy.get_param("~fps", 50)
        self.render = rospy.get_param("~render", False)

        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        rospy.loginfo(
            "Pareto Topology Optimizer initialized | workers=%d, transition_steps=%d, constraint_steps=%d, sustain_steps=%d",
            self.max_workers,
            self.transition_steps,
            self.constraint_steps,
            self.sustain_steps,
        )
        rospy.loginfo(
            "Energy model: %s | loop-specific residuals disabled | evaluation fps=%s",
            OPTIMIZER_ENERGY_MODE,
            self.fps,
        )

    def topo_callback(self, msg):
        n = len(msg.graphs)
        rospy.loginfo("Received %d topologies to process", n)

        if n == 0:
            rospy.logwarn("Empty topology list received")
            return

        try:
            resp = self.get_joints_srv()
            q0 = np.array(real2sim(resp.pos), dtype=float).reshape(-1)
            if q0.shape[0] != 17:
                raise ValueError(
                    "real2sim(get_pos) must return 17 values, got %d" % q0.shape[0]
                )

            rospy.loginfo(
                "Current robot pose acquired (17D). First 5 joints: %s", q0[:5]
            )
        except Exception as e:
            rospy.logerr("Failed to get current joints: %s", str(e))
            return

        tasks = []
        for i, topo in enumerate(msg.graphs):
            processor = TopologyProcessor(
                q0_17=q0,
                topo=topo,
                kinematic_service=self.kinematic_interpreter_srv,
                transition_steps=self.transition_steps,
                constraint_steps=self.constraint_steps,
                sustain_steps=self.sustain_steps,
                fps=self.fps,
                render=self.render,
                index=i,
            )
            tasks.append(self.executor.submit(processor.process))

        min_energy = float("inf")
        optimal_topo = None
        success_count = 0
        fail_count = 0

        for future in as_completed(tasks):
            try:
                processor = future.result()
                if processor.exception:
                    fail_count += 1
                    continue

                if processor.energy is not None and processor.energy < min_energy:
                    min_energy = processor.energy
                    optimal_topo = processor.topo
                success_count += 1
            except Exception as e:
                rospy.logerr("Unexpected error in task: %s", str(e))
                fail_count += 1

        rospy.loginfo(
            "Processing complete - Success: %d, Failed: %d", success_count, fail_count
        )

        if optimal_topo is not None:
            self.opt_pub.publish(optimal_topo)
            rospy.loginfo("Published optimal topology with energy: %.4f J", min_energy)
        else:
            rospy.logwarn("No valid topology found")

    def shutdown(self):
        self.executor.shutdown(wait=False)
        rospy.loginfo("Thread pool shut down.")


if __name__ == "__main__":
    rospy.init_node("pareto_topology_optimizer_node")
    node = ParetoTopologyOptimizerNode()
    try:
        rospy.spin()
    finally:
        node.shutdown()
