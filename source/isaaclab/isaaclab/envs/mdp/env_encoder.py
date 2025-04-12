import torch
import torch.nn as nn
import torch.nn.functional as F

class EnvEncoderMLP(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=256, output_dim=3):
        super(EnvEncoderMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        z = self.fc3(x)
        return z
