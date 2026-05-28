import torch
from torch import nn
from torch.nn import functional as F



class RewardModel(nn.Module):
  def __init__(self, belief_size, state_size, hidden_size, non_linearity='relu'):
    super().__init__()
    self.act_fn = getattr(F, non_linearity)
    self.hidden_size= hidden_size
    self.fc1 = nn.Linear(belief_size+state_size,hidden_size)
    self.fc2 = nn.Linear(hidden_size,hidden_size)
    self.fc3 = nn.Linear(hidden_size,1)
  def forward(self,belief,state):
    hidden = self.act_fn(self.fc1(torch.concat((belief,state),dim=1)))
    hidden = self.act_fn(self.fc2(hidden))
    reward = self.fc3(hidden).squeeze(dim=1)
    return reward