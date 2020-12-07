import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from Param import device


class FocalLoss(nn.Module):

    def __init__(self, gamma = 2):
        super(FocalLoss, self).__init__()
        self.gamma = torch.tensor(gamma).to(device)
        self.alpha = 1.0

    def forward(self, input, target):
        # input are not the probabilities, they are just the cnn out vector
        # input and target shape: (bs, n_classes)
        # sigmoid
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**float(self.gamma) * BCE_loss
        
        return F_loss.mean() #, bce_loss
