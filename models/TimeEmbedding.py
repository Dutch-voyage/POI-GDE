import math
import torch
import torch.nn as nn
from models.Gelu import Gelu
from models.EmbeddingLayer import EmbeddingLayer

class TimeEmbedding(nn.Module):
    def __init__(self, Msup, Minf, Wsup, Winf, Dsup, Dinf, Hsup, Hinf, embed_dim, device):
        super(TimeEmbedding, self).__init__()
        self.device = device
        self.Embeds = nn.Embedding(2, embedding_dim=embed_dim)
        self.Msup = torch.FloatTensor([Msup]).to(self.device)
        self.Minf = torch.FloatTensor([Minf]).to(self.device)
        self.Wsup = torch.FloatTensor([Wsup]).to(self.device)
        self.Winf = torch.FloatTensor([Winf]).to(self.device)
        self.Dsup = torch.FloatTensor([Dsup]).to(self.device)
        self.Dinf = torch.FloatTensor([Dinf]).to(self.device)
        self.Hsup = torch.FloatTensor([Hsup]).to(self.device)
        self.Hinf = torch.FloatTensor([Hinf]).to(self.device)

        self.proj = nn.Sequential(
            nn.Linear(4 * embed_dim, embed_dim),
            nn.GELU(),
        )
        nn.init.kaiming_normal_(self.Embeds.weight)

    def forward(self, timestamp):
        *shape, _ = timestamp.size()
        timestamp = timestamp.view(-1, 5)

        month = timestamp[:, 1]
        week = timestamp[:, 2]
        day = timestamp[:, 3]
        hour = timestamp[:, 4]

        month_embed = ((self.Msup - month).unsqueeze(-1) * self.Embeds(torch.zeros_like(month).long().to(self.device)) +
                        (month - self.Minf).unsqueeze(-1) * self.Embeds(torch.ones_like(month).long().to(self.device))) / \
                          (self.Minf - self.Msup).unsqueeze(-1)
        week_embed = ((self.Wsup - week).unsqueeze(-1) * self.Embeds(torch.zeros_like(week).long().to(self.device)) +
                      (week - self.Winf).unsqueeze(-1) * self.Embeds(torch.ones_like(week).long().to(self.device))) / \
                     (self.Winf - self.Wsup).unsqueeze(-1)
        day_embed = ((self.Dsup - day).unsqueeze(-1) * self.Embeds(torch.zeros_like(day).long().to(self.device)) +
                     (day - self.Dinf).unsqueeze(-1) * self.Embeds(torch.ones_like(day).long().to(self.device))) / \
                    (self.Dinf - self.Dsup).unsqueeze(-1)
        hour_embed = ((self.Hsup - hour).unsqueeze(-1) * self.Embeds(torch.zeros_like(hour).long().to(self.device)) +
                      (hour - self.Hinf).unsqueeze(-1) * self.Embeds(torch.ones_like(hour).long().to(self.device))) / \
                     (self.Hinf - self.Hsup).unsqueeze(-1)
        embed = torch.cat([month_embed, week_embed, day_embed, hour_embed], dim=-1)
        embed = self.proj(embed)
        embed = embed.view(*shape, -1)
        return embed

