import torch
import torch.nn as nn


class RobotConfig:
    def __init__(self, nodes, edges, adjacency_list_aft, external_input):
        # 初始化 nodes, edges, adjacency_list_aft 为可训练的参数

        edges = [row[1:] for row in edges]

        self.nodes = nn.Parameter(torch.tensor(nodes, dtype=torch.float32))  # 8x7
        self.edges = nn.Parameter(torch.tensor(edges, dtype=torch.float32))  # 8x7
        self.adjacency_list_aft = nn.Parameter(
            torch.tensor(adjacency_list_aft, dtype=torch.float32)
        )  # n x 2

        # 外界输入（如长宽高和任务类型）
        self.external_input = torch.tensor(external_input, dtype=torch.float32)

    def get_data(self):
        return self.nodes, self.edges, self.adjacency_list_aft, self.external_input
