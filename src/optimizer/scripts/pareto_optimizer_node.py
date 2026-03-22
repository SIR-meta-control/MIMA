#!/usr/bin/env python3
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rospy

from meta_msgs.msg import TopoList, TopologicalGraph
from meta_msgs.srv import Topo2Angles
from dynamixel_msgs.srv import GetPos

from sim_env import CrimsonMujocoEnv
from utils.sim2real import real2sim


def constrain_judge(pos):
    """17D约束检查：phi_1=phi_2 且 phi+phi_1+phi_2=0。"""
    phi_1 = pos[1]
    phi_2 = pos[2]
    phi = pos[4]
    return np.isclose(phi_1, phi_2, atol=1e-03) and np.isclose(
        phi + phi_1 + phi_2, 0, atol=1e-03
    )


def constrain_planner(current_pos, end_pos, num_steps=100):
    """在主体路径前插入约束修正段（17D向量）。"""

    def _calculate_constrained_target(pos_to_fix):
        target = pos_to_fix.copy()
        phi_1_val = pos_to_fix[1]
        phi_2_val = pos_to_fix[2]

        phi_1_and_2_target = (phi_1_val + phi_2_val) / 2.0
        phi_target = -2.0 * phi_1_and_2_target

        target[1] = phi_1_and_2_target
        target[2] = phi_1_and_2_target
        target[4] = phi_target
        return target

    is_current_constrained = constrain_judge(current_pos)
    is_end_constrained = constrain_judge(end_pos)

    if is_current_constrained and is_end_constrained:
        return None
    elif is_current_constrained and not is_end_constrained:
        constrained_end_pos = _calculate_constrained_target(end_pos)
        return np.linspace(
            start=current_pos, stop=constrained_end_pos, num=num_steps, axis=0
        )
    elif not is_current_constrained and is_end_constrained:
        constrained_start_pos = _calculate_constrained_target(current_pos)
        return np.linspace(
            start=current_pos, stop=constrained_start_pos, num=num_steps, axis=0
        )
    else:
        intermediate_pos = _calculate_constrained_target(current_pos)
        constrained_end_pos = _calculate_constrained_target(end_pos)
        trajectory_1 = np.linspace(
            start=current_pos, stop=intermediate_pos, num=num_steps, axis=0
        )
        trajectory_2 = np.linspace(
            start=intermediate_pos, stop=constrained_end_pos, num=num_steps, axis=0
        )
        return np.vstack((trajectory_1, trajectory_2))


def torque_to_energy(torque, voltage=12, fps=50):
    """扭矩到每帧能耗（与Pareto_extraction同公式，取绝对值）。"""
    current = (
        (0.000130565974) * torque**4
        + (-0.00188139351) * torque**3
        + (0.0216771226) * torque**2
        + (0.410017411) * torque
        + 0.0357777777778
    )
    return np.abs(current * voltage * (1.0 / fps))


def build_trajectory(q0_17, qt_17, transition_steps=400, constraint_steps=100):
    """构建17D轨迹：先约束修正，再主插值。"""
    q0 = q0_17.reshape(-1)
    qt = qt_17.reshape(-1)

    if q0.shape[0] != 17 or qt.shape[0] != 17:
        raise ValueError("build_trajectory requires 17D q0 and qt.")

    main_path = np.linspace(q0, qt, transition_steps, axis=0)
    added_path = constrain_planner(q0, main_path[0], num_steps=constraint_steps)

    if isinstance(added_path, np.ndarray):
        return np.vstack((added_path, main_path))
    return main_path


def compute_energy_for_trajectory(trajectory, sustain_steps=500, fps=50, render=False):
    """
    独立env计算总能耗（变形+维持）。
    并发安全策略：每个任务单独创建并关闭env。
    """
    env = CrimsonMujocoEnv(render_mode=None)
    try:
        energy_trans = 0.0
        for pose in trajectory:
            env.data.ctrl[:] = pose
            env.do_simulation(env.data.ctrl, env.frame_skip)
            joint_torques = env.get_joint_torque()
            energy_trans += np.sum(torque_to_energy(joint_torques, fps=fps))
            if env.render_mode == "human":
                env.render()

        final_pose = trajectory[-1]
        energy_sustain = 0.0
        for _ in range(sustain_steps):
            env.data.ctrl[:] = final_pose
            env.do_simulation(env.data.ctrl, env.frame_skip)
            joint_torques = env.get_joint_torque()
            energy_sustain += np.sum(torque_to_energy(joint_torques, fps=fps))
            if env.render_mode == "human":
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
