import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Attn import AttnBlock
from models.Gelu import Gelu
from models.MultiFuser import MultiFuser
from models.transpose import transpose


class Predictor(nn.Module):
    def __init__(self, embed, number_of_features):
        super(Predictor, self).__init__()
        self.embed = embed
        self.number_of_features = number_of_features
        self.MultiFuser = MultiFuser(embed, number_of_features)
        self.linear = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(embed * number_of_features, embed),
            Gelu()
        )

    def forward(self, x):
        B, C, L, E = x.shape

        h = x.reshape(B * C * L, E, 1)
        h = self.MultiFuser(h)
        h = h.reshape(B, C, L, E)
        h = self.linear(h)

        return h
