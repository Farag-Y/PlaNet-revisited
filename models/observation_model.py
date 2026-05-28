import torch
from torch import nn
from torch.nn import functional as F



class ObservationModel(nn.Module):
  def __init__(self, belief_size, state_size, embedding_size, non_linearity='relu'):
    super().__init__()
    self.act_fn = getattr(F, non_linearity)
    self.embedding_size= embedding_size
    self.fc1 = nn.Linear(belief_size+state_size,embedding_size)
    self.conv1 = nn.ConvTranspose2d(embedding_size,128,5,stride=2)
    self.conv2 = nn.ConvTranspose2d(128,64,5,stride=2)
    self.conv3 = nn.ConvTranspose2d(64,32,6,stride=2)
    self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

  def forward(self,belief,state):
    hidden = self.fc1(torch.cat([belief, state], dim=1))  
    hidden = hidden.view(-1, self.embedding_size, 1, 1) ##TODO: Why reshape is necessary ? dropping batch size ?
    hidden = self.act_fn(self.conv1(hidden))
    hidden = self.act_fn(self.conv2(hidden))
    hidden = self.act_fn(self.conv3(hidden))
    observation = self.conv4(hidden)
    return observation

