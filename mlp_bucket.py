import torch
import os
import torch.nn as nn

class MLP_Bucket(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_pass = nn.Sequential(
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 3),
            nn.Softmax(dim=1))
            
    def forward(self, x):
        x = x.float()
        x = self.first_pass(x)
        x = x.squeeze()
        return x