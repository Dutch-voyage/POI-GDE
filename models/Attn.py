import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transpose import transpose

class AttnBlock(nn.Module):
    def __init__(self, in_ch, dropout=0.1):
        super().__init__()
        self.in_ch = in_ch
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.Sequential(
            nn.BatchNorm1d(in_ch),
        )

        self.proj_q = nn.Conv1d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv1d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv1d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, 1, stride=1, padding=0),
            nn.GELU()
        )
        nn.init.kaiming_normal_(self.proj_q.weight)
        nn.init.kaiming_normal_(self.proj_k.weight)
        nn.init.kaiming_normal_(self.proj_v.weight)

    def forward(self, x):
        B, C, E = x.shape  # [B, C, E]
        # [B, C, E]
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)
        q = q.permute(0, 2, 1)  # [B, E, C]
        # k [B, C, E]
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, E, E]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 1)  # [B, E, C]
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, E, C]
        h = h.permute(0, 2, 1)  # [B, C, E]
        h = self.proj(h)
        # [B, E, C]
        return x + h