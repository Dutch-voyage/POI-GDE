import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class Differentiate(nn.Module):
    def __init__(self, embed, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.Q_weight = nn.Parameter(torch.FloatTensor(3, embed, embed))
        self.Q_bias = nn.Parameter(torch.FloatTensor(3, embed))
        self.V_weight = nn.Parameter(torch.FloatTensor(3, embed, embed))
        self.V_bias = nn.Parameter(torch.FloatTensor(3, embed))
        self.K_weight = nn.Parameter(torch.FloatTensor(3, embed, embed))
        self.K_bias = nn.Parameter(torch.FloatTensor(3, embed))
        self.T_weight = nn.Parameter(torch.FloatTensor(3, embed, embed))
        self.T_bias = nn.Parameter(torch.FloatTensor(3, embed))
        nn.init.xavier_uniform_(self.Q_weight)
        nn.init.xavier_uniform_(self.V_weight)

        nn.init.xavier_uniform_(self.K_weight)
        nn.init.xavier_uniform_(self.T_weight)
        nn.init.zeros_(self.Q_bias)
        nn.init.zeros_(self.V_bias)
        nn.init.zeros_(self.K_bias)
        nn.init.zeros_(self.T_bias)

    def forward(self, A0, x0, u0, t0, t1):
        t0 = t0.unsqueeze(1)
        t1 = t1.unsqueeze(1)

        T0 = (t0.unsqueeze(-2) @ self.T_weight.unsqueeze(0)).squeeze(-2)  # + self.T_bias.unsqueeze(0)
        T1 = (t1.unsqueeze(-2) @ self.T_weight.unsqueeze(0)).squeeze(-2)  # + self.T_bias.unsqueeze(0)
        T0 = T0.unsqueeze(-2)
        T1 = T1.unsqueeze(-2)

        Q_weight = self.Q_weight * T0
        V_weight = self.V_weight * T1
        K_weight = self.K_weight * T1

        Q = (x0[:, :, -1, :].unsqueeze(-2) @ Q_weight).squeeze(-2) + self.Q_bias
        V = (x0.unsqueeze(-2) @ V_weight.unsqueeze(-3)).squeeze(-2) + self.V_bias.unsqueeze(-2)
        K = (u0.unsqueeze(-2) @ K_weight).squeeze(-2) + self.K_bias

        QK = Q * K
        QKV = torch.einsum("bce, bcle -> bcl", QK, V)
        QKV = F.softmax(QKV, dim=-1)

        A1 = torch.zeros_like(A0)
        A1[:, :, :-1, :] = A0[:, :, 1:, :] - A0[:, :, 0, :].unsqueeze(2)
        A1[:, :, -1, :] = A1[:, :, -2, :] + QKV

        w = (T0 @ T1.transpose(-1, -2)).squeeze(-1) / \
            (((T0 @ T0.transpose(-1, -2)).squeeze(-1) + (T1 @ T1.transpose(-1, -2)).squeeze(-1)) ** 0.5)

        x1 = (A1 @ x0)

        x1 = torch.diff(x1, dim=-2, prepend=torch.zeros_like(x1[:, :, 0, :]).unsqueeze(2))

        u1 = u0 * (1 - w) + x1.mean(-2) * w

        return A1 - A0, x1 - x0, u1 - u0