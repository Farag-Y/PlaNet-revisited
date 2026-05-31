from torch import nn
from torch.nn import functional as F



class Encoder(nn.Module):
  def __init__(self, embedding_size, non_linearity='relu'):
    super().__init__()
    self.act_fn = getattr(F, non_linearity)
    self.embedding_size= embedding_size
    #TODO: Calculate input shapes and output shapes to check, spend some time to understand better conv layers and the whole process
    self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
    self.conv2 = nn.Conv2d(32,64,4,stride=2)
    self.conv3 = nn.Conv2d(64,128,4,stride=2)
    self.conv4 = nn.Conv2d(128,256,4,stride=2)
    self.fc1 = nn.Linear(1024,embedding_size)


  def forward(self,observation):
    hidden = self.act_fn(self.conv1(observation))
    hidden = self.act_fn(self.conv2(hidden))
    hidden = self.act_fn(self.conv3(hidden))
    hidden = self.act_fn(self.conv4(hidden))
    hidden = self.fc1(hidden.view(-1,1024))
    return hidden
