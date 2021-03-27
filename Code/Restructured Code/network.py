import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardActorNN(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(FeedForwardActorNN, self).__init__()

        self.layer1 = nn.Linear(in_dim,64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self,obs):
        if isinstance(obs,np.ndarray):
            obs = torch.tensor(obs,dtype=torch.float)

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = torch.softmax(self.layer3(activation2),dim=0)

        return output

class FeedForwardCriticNN(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(FeedForwardCriticNN, self).__init__()

        self.layer1 = nn.Linear(in_dim,64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self,obs):
        if isinstance(obs,np.ndarray):
            obs = torch.tensor(obs,dtype=torch.float)

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        return output