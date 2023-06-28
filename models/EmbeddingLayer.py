import torch
import torch.nn as nn
class EmbeddingLayer(nn.Module):
    def __init__(self, n_item, embed_dim):
        super(EmbeddingLayer, self).__init__()
        self.embeds = nn.Embedding(n_item, embed_dim)
        nn.init.kaiming_normal_(self.embeds.weight)

    def forward(self, idx):
        return self.embeds(idx)