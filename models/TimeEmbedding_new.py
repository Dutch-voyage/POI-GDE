import math
import torch
import torch.nn as nn
from models.Gelu import Gelu
from models.EmbeddingLayer import EmbeddingLayer

class TimeEmbedding(nn.Module):
    def __init__(self, sups, infs, freq_num, embed_dim, device):
        super(TimeEmbedding, self).__init__()
        self.device = device
        self.freq_num = freq_num
        self.sups = torch.FloatTensor(sups).to(self.device)
        self.infs = torch.FloatTensor(infs).to(self.device)
        # self.MEmbeds = nn.Embedding(2, embedding_dim=embed_dim)
        # self.WEmbeds = nn.Embedding(2, embedding_dim=embed_dim)
        # self.DEmbeds = nn.Embedding(2, embedding_dim=embed_dim)
        # self.HEmbeds = nn.Embedding(2, embedding_dim=embed_dim)
        self.Embeds = nn.Embedding(2, embedding_dim=embed_dim)

        self.proj = nn.Sequential(
            nn.Linear(self.freq_num * embed_dim, embed_dim),
            nn.GELU(),
        )
        nn.init.kaiming_normal_(self.Embeds.weight)
        # nn.init.kaiming_normal_(self.MEmbeds.weight)
        # nn.init.kaiming_normal_(self.WEmbeds.weight)
        # nn.init.kaiming_normal_(self.DEmbeds.weight)
        # nn.init.kaiming_normal_(self.HEmbeds.weight)


    def forward(self, timestamp):
        *shape, _ = timestamp.size()
        timestamp = timestamp.view(-1, self.freq_num)
        embeds = []
        for i in range(self.freq_num):
            value = timestamp[:, i]
            embed = ((self.sups[i] - value).unsqueeze(-1) * self.Embeds(torch.zeros_like(value).long()).to(self.device) +
                    (value - self.infs[i]).unsqueeze(-1) * self.Embeds(torch.ones_like(value).long()).to(self.device)) / \
                    (self.sups[i] - self.infs[i]).unsqueeze(-1)
            embeds.append(embed)

        embed = torch.cat(embeds, dim=-1)
        embed = self.proj(embed)
        embed = embed.view(*shape, -1)
        return embed

