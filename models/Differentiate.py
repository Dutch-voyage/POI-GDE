import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class Differentiate(nn.Module):
    def __init__(self, embed):
        super().__init__()
        self.Q = nn.Linear(embed, embed)
        self.V = nn.Linear(embed, embed)
        self.T = nn.Linear(embed, embed)
        self.K = nn.Linear(embed, embed)

    def forward(self, A0, x0, u0, t0, t1):
        Q = self.Q(x0[:, :, -1, :])
        V = self.V(x0)
        T0 = self.T(t0).unsqueeze(1)
        T1 = self.T(t1).unsqueeze(1)
        K = self.K(u0)
        QT = Q * T0
        KT = K * T1
        QTK = QT * K
        VT = V * T1.unsqueeze(1)
        # VTK = VT * K.unsqueeze(-2)

        QTKVT = torch.einsum("bce, bcle -> bcl", QTK, VT)
        QTKVT = F.softmax(QTKVT, dim=-1)

        A1 = torch.zeros_like(A0)
        A1[:, :, :-1, :] = A0[:, :, 1:, :] - A0[:, :, 0, :].unsqueeze(2)
        A1[:, :, -1, :] = A1[:, :, -2, :] + QTKVT


        w = (T0 @ T1.transpose(-1, -2)).squeeze(-1) / \
            (((T0 @ T0.transpose(-1, -2)).squeeze(-1) + (T1 @ T1.transpose(-1, -2)).squeeze(-1)) ** 0.5)
        w = w.unsqueeze(-1)

        x1 = (A1 @ x0)

        x1 = torch.diff(x1, dim=-2, prepend=torch.zeros_like(x1[:, :, 0, :]).unsqueeze(2))

        # attn_score = torch.einsum("bce, bcle -> bcl", KT, VT)
        # attn_score = torch.softmax(attn_score, dim=-1)
        # u1 = (attn_score.unsqueeze(-2) @ x0).squeeze(-2)
        u1 = u0 * (1 - w) + x1.mean(-2) * w

        return A1 - A0, x1 - x0, u1 - u0