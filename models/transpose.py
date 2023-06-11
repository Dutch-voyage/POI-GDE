import torch.nn as nn

class transpose(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)
