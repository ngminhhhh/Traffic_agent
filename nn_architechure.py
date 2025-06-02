import torch
import torch.nn as nn
import torch.nn.functional as F

import random
from collections import deque

class FCNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: list, action_dim: int, device):
        super(FCNetwork, self).__init__()
        self.device = device
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.layers.append(nn.Linear(state_dim, hidden_dim[0]))
        self.bns.append(nn.BatchNorm1d(hidden_dim[0]))

        for i in range(1, len(hidden_dim)):
            self.layers.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
            self.bns.append(nn.BatchNorm1d(hidden_dim[i]))

        self.out = nn.Linear(hidden_dim[-1], action_dim)

        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        for layer, bn in zip(self.layers, self.bns):
            x = layer(x)
            x = bn(x)
            x = F.relu(x)
        q_values = self.out(x)
        return q_values

    
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )
    
    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

        