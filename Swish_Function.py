import torch
import torch.nn as nn




class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


"""
class Swish(nn.Module):
  def forward(self, input):
    return (input * torch.sigmoid(input))
  
  def __repr__(self):
    return self.__class__.__name__ + ' ()'
"""
