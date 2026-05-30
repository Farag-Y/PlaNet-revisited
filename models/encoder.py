from torch import nn
from torch.nn import functional as F



class Encoder(nn.Module):
  def __init__(self, embedding_size, non_linearity='relu'):
    super().__init__()
    self.act_fn = getattr(F, non_linearity)
    self.embedding_size= embedding_size

    self.conv1 = nn.Conv2d(3, 32, 6, stride=2)
    self.conv2 = nn.Conv2d(32,64,6,stride=2)
    self.conv3 = nn.Conv2d(64,128,5,stride=2)
    self.conv4 = nn.Conv2d(128,256,5,stride=2)
    self.fc1 = nn.Linear(1024,embedding_size)


  def forward(self,observation):
    hidden = self.act_fn(self.conv1(observation))
    hidden = self.act_fn(self.conv2(observation))
    hidden = self.act_fn(self.conv3(observation))
    hidden = self.act_fn(self.conv4(observation))
    hidden = self.fc1(hidden) # Is a reshape needed here ?
    return hidden
