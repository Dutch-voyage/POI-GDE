import torch.nn as nn
from models.Attn import AttnBlock
from models.Gelu import Gelu


class MultiFuser(nn.Module):
    def __init__(self, embed, number_of_features):
        super(MultiFuser, self).__init__()
        self.embed = embed
        self.attn = AttnBlock(embed * number_of_features)

    def forward(self, x):
        h = self.attn(x).squeeze(-1)
        return h
