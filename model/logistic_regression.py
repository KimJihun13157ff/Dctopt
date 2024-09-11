import torch
import torch.nn as nn
import torch.optim as optim



class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.linear(x)).squeeze().float()
