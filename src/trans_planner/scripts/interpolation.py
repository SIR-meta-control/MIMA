import numpy as np


def linear_interpolation(start, end, steps):
    """
    线性插值生成关节角度序列。
    :param start: 初始关节角度，1维数组
    :param end: 目标关节角度，1维数组
    :param steps: 插值步数
    :return: 插值后的关节角度序列，2D数组，每行是一个时间步的关节角度
    """

    q0 = np.array(start)
    qt = np.array(end)
    print("q0:", q0)
    print("qt:", qt)
    interp_q = np.linspace(q0, qt, steps)
    return interp_q


def is_aligned(qc):
    """
    检查头尾轴线是否已对齐。
    :param qc: 当前关节角度，1维数组
    :return: 是否对齐
    """
    return np.isclose(qc[1], qc[2], atol=1e-03) and np.isclose(
        qc[4], -(qc[1] + qc[2]), atol=1e-03
    )


class Interpolation:
    def __init__(self, flag_sim, q0, qt, steps=100):
        """
        初始化Planner类。

        :param q0: 初始关节角度，躯干5个，腿部12个，1x17 list:
                    [theta, phi_1, phi_2, theta, phi,
                    qlfr_1, qlfr_2, qlfr_3,
                    qlfl_1, qlfl_2, qlfl_3,
                    qlbr_1, qlbr_2, qlbr_3,
                    qlbl_1, qlbl_2, qlbl_3]
        :param qt: 目标关节角度，躯干5个，腿部12个，1x17 list:
                    [theta, phi_1, phi_2, theta, phi,
                    qlfr_1, qlfr_2, qlfr_3,
                    qlfl_1, qlfl_2, qlfl_3,
                    qlbr_1, qlbr_2, qlbr_3,
                    qlbl_1, qlbl_2, qlbl_3]
        """
        if flag_sim:
            # 原始 list q0 中要删除的元素的索引
            indices_to_remove = {2, 5}  # 使用 set 查找效率更高

            # enumerate(q0) 会生成 (索引, 元素) 的配对
            filtered_q0 = [
                element for i, element in enumerate(q0) if i not in indices_to_remove
            ]
            # 删除躯干上的两个非独立关节角度
            self.q0 = filtered_q0  # 规划起始独立关节角度，一个 list
        else:
            self.q0 = q0  # 规划起始独立关节角度，1x17 list

        self.qt = qt  # 规划目标独立关节角度，1x17 list

        self.steps = steps

    def align_axis(self, qc):
        """
        对齐头尾的轴。
        生成初始姿态到轴对齐的关节角度时间序列轨迹。
        对其轴的关节角度约束：
        1. phi_1 = phi_2
        2. phi + phi_1 + phi_2 = 0

        :param qc: 初始关节角度，1维数组
        :return: 对齐后的关节角度，1维数组
        """
        phi_1_current = qc[1]
        phi_2_current = qc[2]

        phi_1_target = (phi_1_current + phi_2_current) / 2
        phi_2_target = (phi_1_current + phi_2_current) / 2
        phi_target = -(phi_1_target + phi_2_target)

        return np.array([qc[0], phi_1_target, phi_2_target, qc[0], phi_target] + list(qc[5:]))

    def gen_trajectory(self):
        """
        生成关节角度时间序列轨迹。
        根据初始关节角度和目标关节角度，生成一个插值的关节角度序列。
        轨迹生成规则：
        1. 如果初始和目标都已对齐，直接插值生成轨迹。
        2. 如果初始已对齐但目标未对齐，插值进入theta=pi或theta=0空间再插值到目标。
        3. 如果初始未对齐但目标已对齐，插值到轴对齐再插值到目标。
        4. 如果初始和目标都未对齐，先插值到轴对齐再插值进入theta=pi或theta=0空间再插值到目标。
        :return: 关节角度时间序列轨迹，2D数组，每行是一个时间步的关节角度
        """
        # 判断当前是否已对齐
        is_initial_aligned = is_aligned(self.q0)
        is_target_aligned = is_aligned(self.qt)

        # 如果初始和目标都已对齐，直接插值生成轨迹
        if is_initial_aligned and is_target_aligned:
            interp_q = linear_interpolation(self.q0, self.qt, self.steps)

        # 如果初始已对齐但目标未对齐，插值进入theta=pi或theta=0空间再插值到目标
        elif is_initial_aligned and not is_target_aligned:
            qtheta = self.align_axis(self.qt)
            interp_q_0 = linear_interpolation(self.q0, qtheta, self.steps // 2)
            interp_q_1 = linear_interpolation(qtheta, self.qt, self.steps // 2)
            interp_q = np.vstack((interp_q_0, interp_q_1))

        # 如果初始未对齐但目标已对齐，插值到轴对齐再插值到目标
        elif not is_initial_aligned and is_target_aligned:
            qaligned = self.align_axis(self.q0)
            interp_q_0 = linear_interpolation(self.q0, qaligned, self.steps // 2)
            interp_q_1 = linear_interpolation(qaligned, self.qt, self.steps // 2)
            interp_q = np.vstack((interp_q_0, interp_q_1))

        # 如果初始和目标都未对齐，先插值到轴对齐再插值进入theta=pi或theta=0空间再插值到目标
        elif not is_initial_aligned and not is_target_aligned:
            qaligned = self.align_axis(self.q0)
            qtheta = self.align_axis(self.qt)
            interp_q_0 = linear_interpolation(self.q0, qaligned, self.steps // 3)
            interp_q_1 = linear_interpolation(qaligned, qtheta, self.steps // 3)
            interp_q_2 = linear_interpolation(qtheta, self.qt, self.steps // 3)
            interp_q = np.vstack((interp_q_0, interp_q_1, interp_q_2))

        return interp_q
