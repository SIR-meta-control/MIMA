import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import random
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch


class RobotConfigurationNet(nn.Module):
    def __init__(self, args, num_configs=3, latent_dim=32, leg_latent_dim=8):
        super(RobotConfigurationNet, self).__init__()

        # Store args and batch_size
        self.args = args
        self.batch_size = args.batch_size

        # Load graph imputation data
        graph_imputation = np.load(args.graph_imputation_path, allow_pickle=True).item()
        self.T_node_leg = torch.tensor(
            graph_imputation["T_node_leg"], dtype=torch.float32, device="cuda"
        )
        self.T_node_edge = torch.tensor(
            graph_imputation["T_node_edge"], dtype=torch.float32, device="cuda"
        )
        self.S_edge_spacing = torch.tensor(
            graph_imputation["S_edge_spacing"], dtype=torch.float32, device="cuda"
        )

        self.num_configs = num_configs
        self.latent_dim = latent_dim
        self.leg_latent_dim = leg_latent_dim

        # Encoder - 添加bar信息 (3维one-hot编码)
        self.node_encoder_fc1 = nn.Linear(
            8 + 8 * 7 + 3, 256
        )  # Vreq + nodes + bar_onehot
        self.node_encoder_fc2 = nn.Linear(256, 128)
        self.node_encoder_mean = nn.Linear(128, latent_dim)
        self.node_encoder_logvar = nn.Linear(128, latent_dim)

        self.leg_encoder_fc1 = nn.Linear(
            8 + 1 + 3, 64
        )  # Vreq + 可变leg_angle + bar_onehot
        self.leg_encoder_fc2 = nn.Linear(64, 32)
        self.leg_encoder_mean = nn.Linear(32, leg_latent_dim)
        self.leg_encoder_logvar = nn.Linear(32, leg_latent_dim)

        # Decoder - 添加bar信息
        self.node_decoder_fc1 = nn.Linear(
            8 + latent_dim + 3, 128
        )  # Vreq + z + bar_onehot
        self.node_decoder_fc2 = nn.Linear(128, 256)
        self.node_decoder_fc3 = nn.Linear(256, 512)
        self.nodes_out = nn.Linear(512, 8 * 7)

        self.leg_decoder_fc1 = nn.Linear(
            8 + leg_latent_dim + 3, 32
        )  # Vreq + z + bar_onehot
        self.leg_decoder_fc2 = nn.Linear(32, 16)
        self.leg_out = nn.Linear(16, 1)  # 只输出1维可变值

        # GCN
        self.conv1 = GCNConv(7, 128)
        self.conv2 = GCNConv(128, 256)
        self.conv3 = GCNConv(256, 512)

        # scale and confidence
        self.fc4 = nn.Linear(512 + 3, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 3)
        self.fc_confidence = nn.Linear(128, 1)

        # Inference Encoder - 也需要添加bar信息
        self.node_vreq_encoder_fc1 = nn.Linear(8 + 3, 64)  # Vreq + bar_onehot
        self.node_vreq_encoder_fc2 = nn.Linear(64, 128)
        self.node_vreq_encoder_mean = nn.Linear(128, latent_dim)
        self.node_vreq_encoder_logvar = nn.Linear(128, latent_dim)

        self.leg_vreq_encoder_fc1 = nn.Linear(8 + 3, 32)  # Vreq + bar_onehot
        self.leg_vreq_encoder_fc2 = nn.Linear(32, 16)
        self.leg_vreq_encoder_mean = nn.Linear(16, leg_latent_dim)
        self.leg_vreq_encoder_logvar = nn.Linear(16, leg_latent_dim)

    def apply_leg_angle_constraints(self, leg_angle_variable, bar_list):
        """
        根据bar类型应用leg_angle约束

        Args:
            leg_angle_variable: [batch, 1] 可变的leg_angle维度
            bar_list: bar类型列表

        Returns:
            leg_angle: [batch, 3] 完整的3维leg_angle
        """
        batch_size = len(bar_list)
        leg_angle = torch.zeros(batch_size, 3, device=leg_angle_variable.device)

        for b in range(batch_size):
            if bar_list[b] == "6-bar":
                # 6-bar: [可变, -1.95, 1.95]
                leg_angle[b, 0] = leg_angle_variable[b, 0]
                leg_angle[b, 1] = -1.95
                leg_angle[b, 2] = 1.95
            elif bar_list[b] == "4-bar":
                # 4-bar: [可变, -0.251, 1.85]
                leg_angle[b, 0] = leg_angle_variable[b, 0]
                leg_angle[b, 1] = -0.251
                leg_angle[b, 2] = 1.85
            elif bar_list[b] == "8-bar":
                # 8-bar: [0, 可变, 1.95]
                leg_angle[b, 0] = 0.0
                leg_angle[b, 1] = leg_angle_variable[b, 0]
                leg_angle[b, 2] = 1.95

        return leg_angle

    def apply_quaternion_constraints(self, nodes, bar_list):
        """
        对不同类型的机构应用四元数约束

        Args:
            nodes: [batch_size, 8, 7] 节点数据
            bar_list: bar类型列表

        Returns:
            nodes: 应用约束后的节点数据
        """
        batch_size = len(bar_list)
        constrained_nodes = []

        for b in range(batch_size):
            node_b = nodes[b].clone()  # [8, 7]

            if bar_list[b] == "6-bar":
                # 6-bar: 四元数约束为接近 [1, 0, 0, 0]
                # 保持位置不变，只约束四元数
                positions = node_b[:, :3]  # [8, 3]
                quaternions = node_b[:, 3:]  # [8, 4]

                # 设置四元数为接近 [1, 0, 0, 0] 的值
                # w分量至少为0.9，其他分量接近0
                w_component = torch.clamp(quaternions[:, 0], min=0.9)  # 确保w >= 0.9

                # 其他分量设置为很小的值
                x_component = quaternions[:, 1] * 0.1  # 减小x分量
                y_component = quaternions[:, 2] * 0.1  # 减小y分量
                z_component = quaternions[:, 3] * 0.1  # 减小z分量

                # 重新归一化四元数
                new_quaternions = torch.stack(
                    [w_component, x_component, y_component, z_component], dim=1
                )
                norm = torch.norm(new_quaternions, dim=1, keepdim=True)
                normalized_quaternions = new_quaternions / torch.clamp(norm, min=1e-6)

                # 确保w分量仍然大于等于0.9
                w_vals = normalized_quaternions[:, 0]
                mask = w_vals < 0.9
                if torch.any(mask):
                    # 如果归一化后w分量小于0.9，重新设置
                    w_constrained = torch.where(
                        mask, torch.tensor(0.95, device=w_vals.device), w_vals
                    )

                    # 重新计算其他分量以保持归一化
                    remaining_norm_sq = 1.0 - w_constrained**2
                    remaining_norm = torch.sqrt(
                        torch.clamp(remaining_norm_sq, min=1e-6)
                    )

                    other_components = normalized_quaternions[:, 1:4]
                    other_norm = torch.norm(other_components, dim=1, keepdim=True)
                    other_norm = torch.clamp(other_norm, min=1e-6)

                    other_constrained = (
                        other_components / other_norm * remaining_norm.unsqueeze(1)
                    )

                    normalized_quaternions = torch.cat(
                        [w_constrained.unsqueeze(1), other_constrained], dim=1
                    )

                # 组合位置和约束后的四元数
                constrained_node = torch.cat([positions, normalized_quaternions], dim=1)
                constrained_nodes.append(constrained_node)
            else:
                constrained_nodes.append(node_b)

        return torch.stack(constrained_nodes, dim=0)

    def extract_variable_leg_angle(self, leg_angle, bar_list):
        """
        从完整的leg_angle中提取可变维度

        Args:
            leg_angle: [batch, 3] 完整的3维leg_angle
            bar_list: bar类型列表

        Returns:
            leg_angle_variable: [batch, 1] 可变的leg_angle维度
        """
        batch_size = len(bar_list)
        leg_angle_variable = torch.zeros(batch_size, 1, device=leg_angle.device)

        for b in range(batch_size):
            if bar_list[b] == "6-bar":
                # 6-bar: 第0个可变
                leg_angle_variable[b, 0] = leg_angle[b, 0]
            elif bar_list[b] == "4-bar":
                # 4-bar: 第0个可变
                leg_angle_variable[b, 0] = leg_angle[b, 0]
            elif bar_list[b] == "8-bar":
                # 8-bar: 第1个可变
                leg_angle_variable[b, 0] = leg_angle[b, 1]

        return leg_angle_variable

    def bar_to_onehot(self, bar_list):
        """将bar类型转换为one-hot编码"""
        batch_size = len(bar_list)
        bar_onehot = torch.zeros(batch_size, 3)  # [4-bar, 6-bar, 8-bar]

        for i, bar in enumerate(bar_list):
            if bar == "4-bar":
                bar_onehot[i, 0] = 1
            elif bar == "6-bar":
                bar_onehot[i, 1] = 1
            elif bar == "8-bar":
                bar_onehot[i, 2] = 1

        return bar_onehot

    def determine_bar_from_vreq(self, Vreq):
        """根据Vreq中的任务信息确定bar类型"""
        batch_size = Vreq.shape[0]
        bar_list = []

        for b in range(batch_size):
            vreq_b = Vreq[b]
            if vreq_b[5] == 1:  # inspect task
                bar_list.append("4-bar")  # 或者根据你的逻辑调整
            elif vreq_b[6] == 1:  # load task
                bar_list.append("6-bar")
            elif vreq_b[7] == 1:  # pack task
                bar_list.append(random.choice(["4-bar", "8-bar"]))
            else:  # 如果都为0，随机选择
                bar_list.append(random.choice(["4-bar", "6-bar", "8-bar"]))

        return bar_list

    def encode_node(self, Vreq, nodes, bar_onehot):
        batch_size = Vreq.shape[0]
        nodes_flat = nodes.view(batch_size, -1)
        x = torch.cat([Vreq, nodes_flat, bar_onehot], dim=1)

        x = F.relu(self.node_encoder_fc1(x))
        x = F.relu(self.node_encoder_fc2(x))

        mean = self.node_encoder_mean(x)
        logvar = self.node_encoder_logvar(x)

        return mean, logvar

    def encode_leg(self, Vreq, leg_angle, bar_list, bar_onehot):
        batch_size = Vreq.shape[0]
        # 提取可变的leg_angle维度
        leg_angle_variable = self.extract_variable_leg_angle(leg_angle, bar_list)
        leg_angle_flat = leg_angle_variable.view(batch_size, -1)
        x = torch.cat([Vreq, leg_angle_flat, bar_onehot], dim=1)

        x = F.relu(self.leg_encoder_fc1(x))
        x = F.relu(self.leg_encoder_fc2(x))

        mean = self.leg_encoder_mean(x)
        logvar = self.leg_encoder_logvar(x)

        return mean, logvar

    def encode_vreq_node(self, Vreq, bar_onehot):
        x = torch.cat([Vreq, bar_onehot], dim=1)
        x = F.relu(self.node_vreq_encoder_fc1(x))
        x = F.relu(self.node_vreq_encoder_fc2(x))
        mean = self.node_vreq_encoder_mean(x)
        logvar = self.node_vreq_encoder_logvar(x)
        return mean, logvar

    def encode_vreq_leg(self, Vreq, bar_onehot):
        x = torch.cat([Vreq, bar_onehot], dim=1)
        x = F.relu(self.leg_vreq_encoder_fc1(x))
        x = F.relu(self.leg_vreq_encoder_fc2(x))
        mean = self.leg_vreq_encoder_mean(x)
        logvar = self.leg_vreq_encoder_logvar(x)
        return mean, logvar

    def decode_node(self, Vreq, z, bar_onehot, bar_list=None):
        batch_size = Vreq.shape[0]  # Dynamic batch size
        x_nodes = torch.cat([Vreq, z, bar_onehot], dim=1)
        x_nodes = F.relu(self.node_decoder_fc1(x_nodes))
        x_nodes = F.relu(self.node_decoder_fc2(x_nodes))
        x_nodes = F.relu(self.node_decoder_fc3(x_nodes))
        nodes = self.nodes_out(x_nodes).view(batch_size, 8, 7)

        # Quaternion normalization
        quaternions = nodes[:, :, 3:]
        norm = torch.norm(quaternions, dim=-1, keepdim=True)
        normalized_quaternions = quaternions / norm
        nodes = torch.cat([nodes[:, :, :3], normalized_quaternions], dim=2)

        # 应用bar类型特定的四元数约束
        if bar_list is not None:
            nodes = self.apply_quaternion_constraints(nodes, bar_list)

        return nodes

    def decode_leg(self, Vreq, z_leg, bar_list, bar_onehot):
        x = torch.cat([Vreq, z_leg, bar_onehot], dim=1)
        x = F.relu(self.leg_decoder_fc1(x))
        x = F.relu(self.leg_decoder_fc2(x))
        leg_angle_variable = self.leg_out(x)  # [batch, 1]

        # 应用约束，生成完整的3维leg_angle
        leg_angle = self.apply_leg_angle_constraints(leg_angle_variable, bar_list)

        return leg_angle

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, Vreq, nodes=None, leg_angle=None, bar_list=None, train=True):
        if train:
            # 训练模式：使用给定的bar信息
            bar_onehot = self.bar_to_onehot(bar_list).to(Vreq.device)

            # Main VAE encoding with ground truth
            mean_nodes, logvar_nodes = self.encode_node(Vreq, nodes, bar_onehot)
            mean_leg, logvar_leg = self.encode_leg(
                Vreq, leg_angle, bar_list, bar_onehot
            )

            # Condition-only encoding for inference training
            mean_nodes_vreq, logvar_nodes_vreq = self.encode_vreq_node(Vreq, bar_onehot)
            mean_leg_vreq, logvar_leg_vreq = self.encode_vreq_leg(Vreq, bar_onehot)

            z_nodes = self.reparameterize(mean_nodes, logvar_nodes)
            z_leg = self.reparameterize(mean_leg, logvar_leg)

            nodes_pred = self.decode_node(Vreq, z_nodes, bar_onehot, bar_list)
            leg_angle_pred = self.decode_leg(Vreq, z_leg, bar_list, bar_onehot)

            nodes_t = self.convert_to_transformation(nodes_pred)
            edges_t = self.generate_edges_from_nodes(nodes_t)
            leg_base_t = self.generate_legs_from_nodes(nodes_t)
            graph = self.generate_graph_data(Vreq, nodes_pred, bar_list)

            x = graph.x
            edge_index = graph.edge_index
            batch_idx = graph.batch

            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = F.relu(self.conv3(x, edge_index))

            x = global_mean_pool(x, batch_idx)

            x = torch.cat([x, leg_angle_pred], dim=1)

            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))
            scale_pred = self.fc6(x)
            confidence = torch.sigmoid(self.fc_confidence(x))

            edges_pred = self.convert_to_parameter_func(edges_t)
            leg_base_pred = self.convert_to_parameter_func(leg_base_t)

            return (
                nodes_pred,
                edges_pred,
                edges_t,
                leg_base_pred,
                leg_angle_pred,
                scale_pred,
                confidence,
                mean_nodes,
                logvar_nodes,
                mean_leg,
                logvar_leg,
                mean_nodes_vreq,
                logvar_nodes_vreq,
                mean_leg_vreq,
                logvar_leg_vreq,
            )

        else:
            import time

            torch.manual_seed(int(time.time() * 1000) % (2**32))

            # Inference: Generate multiple configurations
            all_configs = []
            all_confidences = []

            for i in range(self.num_configs):
                # 为每个构型随机选择bar类型
                # bar_type = random.choice(["6-bar"])
                bar_type = random.choice(["4-bar", "6-bar", "8-bar"])
                bar_list = [bar_type]  # 单个元素的列表
                bar_onehot = self.bar_to_onehot(bar_list).to(Vreq.device)

                # Use condition-aware inference encoders to get better prior distributions
                mean_nodes_prior, logvar_nodes_prior = self.encode_vreq_node(
                    Vreq, bar_onehot
                )
                mean_leg_prior, logvar_leg_prior = self.encode_vreq_leg(
                    Vreq, bar_onehot
                )

                # Sample from condition-aware prior instead of pure random noise
                z_nodes = self.reparameterize(mean_nodes_prior, logvar_nodes_prior)
                z_leg = self.reparameterize(mean_leg_prior, logvar_leg_prior)

                # Add some additional randomness for diversity
                z_nodes += 0.1 * torch.randn_like(z_nodes)
                z_leg += 0.1 * torch.randn_like(z_leg)

                nodes_pred = self.decode_node(Vreq, z_nodes, bar_onehot, bar_list)
                leg_angle_pred = self.decode_leg(Vreq, z_leg, bar_list, bar_onehot)

                # Use functional methods to generate all data
                nodes_t = self.convert_to_transformation(nodes_pred)
                edges_t = self.generate_edges_from_nodes(nodes_t)
                leg_base_t = self.generate_legs_from_nodes(nodes_t)
                graph = self.generate_graph_data(Vreq, nodes_pred, bar_list)

                x = graph.x
                edge_index = graph.edge_index
                batch_idx = graph.batch

                x = F.relu(self.conv1(x, edge_index))
                x = F.relu(self.conv2(x, edge_index))
                x = F.relu(self.conv3(x, edge_index))

                x = global_mean_pool(x, batch_idx)

                x = torch.cat([x, leg_angle_pred], dim=1)

                x = F.relu(self.fc4(x))
                x = F.relu(self.fc5(x))
                scale_pred = self.fc6(x)
                confidence = torch.sigmoid(self.fc_confidence(x))

                edges_pred = self.convert_to_parameter_func(edges_t)
                leg_base_pred = self.convert_to_parameter_func(leg_base_t)

                config = {
                    "nodes": nodes_pred,
                    "edges": edges_pred,
                    "leg_base": leg_base_pred,
                    "leg_angle": leg_angle_pred,
                    "scale": scale_pred,
                    "bar": bar_list,  # 返回使用的bar类型
                }

                all_configs.append(config)
                all_confidences.append(confidence)

            return all_configs, all_confidences

    def convert_to_transformation(self, lst):
        batch_size, num, _ = lst.shape
        T = torch.zeros((batch_size, num, 4, 4), device=lst.device)

        for b in range(batch_size):
            for n in range(num):
                pos = lst[b, n, :3]
                quat = lst[b, n, 3:]
                w, x, y, z = quat

                R = torch.stack(
                    [
                        torch.stack(
                            [
                                1 - 2 * (y * y + z * z),
                                2 * (x * y - w * z),
                                2 * (x * z + w * y),
                            ]
                        ),
                        torch.stack(
                            [
                                2 * (x * y + w * z),
                                1 - 2 * (x * x + z * z),
                                2 * (y * z - w * x),
                            ]
                        ),
                        torch.stack(
                            [
                                2 * (x * z - w * y),
                                2 * (y * z + w * x),
                                1 - 2 * (x * x + y * y),
                            ]
                        ),
                    ]
                )

                T[b, n, :3, :3] = R
                T[b, n, :3, 3] = pos
                T[b, n, 3, 3] = 1.0

        return T

    def generate_edges_from_nodes(self, nodes_t):
        pair = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 0}
        batch_size, _, row, col = nodes_t.shape
        edges_t = torch.zeros((batch_size, 8, row, col), device=nodes_t.device)

        for b in range(batch_size):
            for n in range(8):
                edges_t[b, n] = torch.matmul(nodes_t[b, pair[n]], self.T_node_edge[n])

        return edges_t

    def generate_legs_from_nodes(self, nodes_t):
        pair = {0: 2, 1: 3, 2: 6, 3: 7}
        batch_size, _, row, col = nodes_t.shape
        leg_base_t = torch.zeros((batch_size, 4, row, col), device=nodes_t.device)

        for b in range(batch_size):
            for n in range(4):
                leg_base_t[b, n] = torch.matmul(nodes_t[b, pair[n]], self.T_node_leg[n])

        return leg_base_t

    def generate_graph_data(self, Vreq, nodes, bars):
        graph_list = []
        batch_size = Vreq.shape[0]

        for b in range(batch_size):
            nodes_b = nodes[b]
            bar_b = bars[b]

            if bar_b == "4-bar":
                edge_index = (
                    torch.tensor([[1, 2], [2, 3], [3, 4]], device=nodes.device)
                    .t()
                    .contiguous()
                )
            elif bar_b == "8-bar":
                edge_index = (
                    torch.tensor(
                        [
                            [0, 1],
                            [1, 2],
                            [2, 3],
                            [3, 4],
                            [4, 5],
                            [5, 6],
                            [6, 7],
                            [7, 0],
                        ],
                        device=nodes.device,
                    )
                    .t()
                    .contiguous()
                )
            elif bar_b == "6-bar":
                edge_index = (
                    torch.tensor(
                        [[1, 2], [2, 3], [3, 4], [4, 6], [6, 7], [7, 1]],
                        device=nodes.device,
                    )
                    .t()
                    .contiguous()
                )

            edges = []
            for i in range(edge_index.size(1)):
                node1_idx, node2_idx = edge_index[0, i].item(), edge_index[1, i].item()
                node1, node2 = nodes_b[node1_idx], nodes_b[node2_idx]

                direction = node2[:3] - node1[:3]
                length = torch.norm(direction)
                direction = direction / length

                edge_feature = torch.cat([direction, length.unsqueeze(0)])
                edges.append(edge_feature)

            edges = torch.stack(edges)

            graph_data = Data(x=nodes_b, edge_index=edge_index, edge_attr=edges)
            graph_list.append(graph_data)

        graph = Batch.from_data_list(graph_list)
        device = nodes.device
        graph.x = graph.x.to(device)
        graph.edge_index = graph.edge_index.to(device)
        graph.batch = graph.batch.to(device)
        if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
            graph.edge_attr = graph.edge_attr.to(device)

        return graph

    def convert_to_parameter_func(self, T):
        batch_size, num, _, _ = T.shape
        lst = torch.zeros((batch_size, num, 7), device=T.device)

        for b in range(batch_size):
            for n in range(num):
                pos = T[b, n, :3, 3]
                R = T[b, n, :3, :3]

                trace = torch.trace(R)
                if trace > 0:
                    s = torch.sqrt(trace + 1.0) * 2  # s = 4 * w
                    w = 0.25 * s
                    x = (R[2, 1] - R[1, 2]) / s
                    y = (R[0, 2] - R[2, 0]) / s
                    z = (R[1, 0] - R[0, 1]) / s
                elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                    s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
                    w = (R[2, 1] - R[1, 2]) / s
                    x = 0.25 * s
                    y = (R[0, 1] + R[1, 0]) / s
                    z = (R[0, 2] + R[2, 0]) / s
                elif R[1, 1] > R[2, 2]:
                    s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
                    w = (R[0, 2] - R[2, 0]) / s
                    x = (R[0, 1] + R[1, 0]) / s
                    y = 0.25 * s
                    z = (R[1, 2] + R[2, 1]) / s
                else:
                    s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
                    w = (R[1, 0] - R[0, 1]) / s
                    x = (R[0, 2] + R[2, 0]) / s
                    y = (R[1, 2] + R[2, 1]) / s
                    z = 0.25 * s

                lst[b, n, :3] = pos
                lst[b, n, 3:] = torch.stack([w, x, y, z])

        return lst


def load_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data
