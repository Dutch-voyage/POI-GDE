import torch
import torch.nn as nn
import torch.nn.functional as F

class GeoEmbedding(nn.Module):
    def __init__(self, Latsup, Latinf, Lonsup, Loninf, embed_dim, device):
        super(GeoEmbedding, self).__init__()
        self.device = device
        self.Embeds = nn.Embedding(2, embedding_dim=embed_dim)
        # self.LatEmbeds = nn.Embedding(2, embedding_dim=embed_dim)
        # self.LonEmbeds = nn.Embedding(2, embedding_dim=embed_dim)
        self.Latsup = torch.FloatTensor([Latsup]).to(self.device)
        self.Latinf = torch.FloatTensor([Latinf]).to(self.device)
        self.Lonsup = torch.FloatTensor([Lonsup]).to(self.device)
        self.Loninf = torch.FloatTensor([Loninf]).to(self.device)
        self.proj = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU()
        )

        nn.init.kaiming_normal_(self.Embeds.weight)
        # nn.init.kaiming_normal_(self.LatEmbeds.weight)
        # nn.init.kaiming_normal_(self.LonEmbeds.weight)

    def forward(self, loc):
        lat = loc[:, 0]
        lon = loc[:, 1]

        lat_embed = ((self.Latsup - lat).unsqueeze(-1) * self.Embeds(torch.zeros_like(lat).long().to(self.device)) +
                     (lat - self.Latinf).unsqueeze(-1) * self.Embeds(torch.ones_like(lat).long().to(self.device))) / \
                    (self.Latinf - self.Latsup).unsqueeze(-1)
        lon_embed = ((self.Lonsup - lon).unsqueeze(-1) * self.Embeds(torch.zeros_like(lon).long().to(self.device)) +
                     (lon - self.Loninf).unsqueeze(-1) * self.Embeds(torch.ones_like(lon).long().to(self.device))) / \
                    (self.Loninf - self.Lonsup).unsqueeze(-1)
        embed = torch.cat([lat_embed, lon_embed], dim=-1)
        embed = self.proj(embed)
        return embed
