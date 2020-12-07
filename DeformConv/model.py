from __future__ import absolute_import, division

import torch
import torch.nn.functional as F
import torch.nn as nn
from DeformConv.layers import ConvOffset2D

from Param import nz, nc

class DeformConvNet_Encoder256(nn.Module):
    def __init__(self, nc=nc, nf=16, nz=nz):
        super(DeformConvNet_Encoder256, self).__init__()
        
        self.nz = nz
        n_features = [nf, nf*2, nf*4, nf*8, nf*16, nf*32, nf*64]

        # first conv 
        self.first_conv = nn.Sequential( nn.Conv2d(nc, n_features[0], 3, 2, 1, bias=False),
                                          nn.BatchNorm2d(nf),
                                          nn.ReLU()
                                        )

        self.conv_layers = nn.ModuleList([ nn.Sequential( ConvOffset2D(n_features[i]),
                                      nn.Conv2d(n_features[i], n_features[i+1], 3, 2, 1, bias=False),
                                              nn.BatchNorm2d(n_features[i+1]),
                                              nn.LeakyReLU(0.2, inplace=True)
                                          ) for i in range(len(n_features)-2)])

        self.conv = nn.Sequential( nn.Conv2d(n_features[-2], n_features[-1], 3, 2, 0, bias=False),
                                          nn.BatchNorm2d(n_features[-1]),
                                          nn.ReLU()
                                        )

        self.out_conv = nn.Sequential( nn.Conv2d(n_features[-1], nz, 1, 1, 0, bias=True),
                                       nn.Sigmoid()
                                     )

    def forward(self, x, verbose=False):
        x = self.first_conv(x)
        if verbose: print(x.shape)

        for layer in self.conv_layers:
            x = layer(x)
            if verbose: print(x.shape)

        x = self.conv(x)
        if verbose: print(x.shape)
 
        x = self.out_conv(x)
        if verbose: print(x.shape)

        return x.view(-1, self.nz)

    def nb_parameters(self):
        count = torch.zeros(1)
        for param in super(DeformConvNet_Encoder256, self).parameters():
           count += torch.prod(torch.tensor(param.size()))
        return int(count.item())



class Decoder256(nn.Module):
    def __init__(self,nz=nz,ngf=16,nc=nc):
        super(Decoder256, self).__init__()
        self.nz=nz
        self.nc=nc
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf *64 , 2, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 64),
            nn.ReLU(True),
            # size ngf*64 x2 x2
            nn.ConvTranspose2d(ngf * 64, ngf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True),
            # size ngf*32 x4 x4
            nn.ConvTranspose2d(ngf*32, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x16
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (nc) x 128 x 128
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            #nn.Sigmoid() # for VAE
            # state size. (nc) x 256 x 256

        )

    def forward(self, input):
        output = self.main(input.reshape(-1, self.nz, 1, 1))
        return output

