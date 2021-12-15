import torch, torch.nn as nn, torch.nn.functional as F

class StatsPool(nn.Module):
    
    def __init__(self):
        super(StatsPool, self).__init__()

    def forward(self, x):
        # input: batch * embd_dim * ...
        x = x.view(x.shape[0], x.shape[1], -1)
        means = x.mean(dim=2)
        stds = torch.sqrt(((x - means.unsqueeze(2))**2).sum(dim=2).clamp(min=1e-8) / (x.shape[2] - 1))
        out = torch.cat([means, stds], dim=1)
        return out

    
class AvgPool(nn.Module):
    
    def __init__(self):
        super(AvgPool, self).__init__()

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        return x.mean(dim=2)
    

class AttentivePool(nn.Module):
    
    def __init__(self, embed_dim, reduction, att_channel):
        super(AttentivePool, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim//reduction)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(embed_dim//reduction, att_channel)
            
    def forward(self, x):
        # input: batch * frame * embd_dim
        weights = F.softmax(self.fc2(self.tanh(self.fc1(x))), dim=1) # batch * frame * att_channel
        out = (weights[:, :, 0].unsqueeze(dim=-1) * x).mean(dim=1)
        for i in range(1, weights.shape[-1]):
            out = torch.cat((out, (weights[:, :, 0].unsqueeze(dim=-1) * x).mean(dim=1)), dim=-1)
        return out