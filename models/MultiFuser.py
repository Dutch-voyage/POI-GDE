import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Attn import AttnBlock
from models.Gelu import Gelu
from models.transpose import transpose


class MultiFuser(nn.Module):
    def __init__(self, embed, number_of_features):
        super(MultiFuser, self).__init__()
        self.embed = embed
        self.attn1 = AttnBlock(embed * number_of_features)
        self.attn2 = AttnBlock(embed * number_of_features)
        self.attn3 = AttnBlock(embed * number_of_features)
        self.attn4 = AttnBlock(embed * number_of_features)
        self.head = nn.Sequential(
            # nn.BatchNorm1d(embed * number_of_features),
            nn.Linear(embed * number_of_features, embed),
            Gelu(),
        )
        self.tail = nn.Sequential(
            nn.BatchNorm1d(embed * number_of_features),
            nn.Linear(embed * number_of_features, embed),
            Gelu(),
        )
        self.linear3 = nn.Sequential(
            # nn.BatchNorm1d(embed * number_of_features),
            nn.Linear(embed * number_of_features, embed),
            Gelu(),
        )

    def forward(self, x):
        h = self.attn1(x).squeeze(-1)
        # h = self.attn2(h).squeeze(-1)
        # h = self.tail(h.squeeze(-1))

        return h
