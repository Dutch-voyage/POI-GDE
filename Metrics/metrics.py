import torch
import torchsort

def hit_rate(input, target, k):
    y_score = torch.gather(input, dim=-1, index=target.unsqueeze(dim=-1))
    rank = torch.gt(input, y_score).sum(dim=-1)
    return (rank < k).float()

def reciprocal_rank(input, target, k):
    y_score = torch.gather(input, dim=-1, index=target.unsqueeze(dim=-1))
    rank = torch.gt(input, y_score).sum(dim=-1)
    score = torch.reciprocal(rank + 1.0)
    if k is not None:
        score[rank >= k] = 0.0
    return score

def threshold(input, low, high):
    return torch.where(input < low, torch.tensor(0.0).to(input.device), torch.where(input > high, high, input - low))

def SortLoss(input, target):

    rank = input.shape[1] - torchsort.soft_rank(input, regularization='kl', regularization_strength=10)
    print(rank)
    # target_rank = torch.gather(rank, dim=-1, index=target.unsqueeze(dim=-1))
    #
    print('min')
    min = rank.min(dim=-1)[0].unsqueeze(dim=1)
    print(min.mean())
    target_rank = torch.gather(rank, dim=-1, index=target.unsqueeze(dim=-1))
    loss = (torch.log(target_rank + 1)).mean()
    # k_low = torch.FloatTensor([k_low]).to(input.device)
    # k_high = torch.FloatTensor([k_high]).to(input.device)
    # loss = threshold(target_rank, k_low, k_high).mean()
    return loss