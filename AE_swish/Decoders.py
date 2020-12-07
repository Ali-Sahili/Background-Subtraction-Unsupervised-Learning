""" models of Generators, Endoders and Discriminators at various image sizes
following deep convolutionnal model of DCGAN
cf https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
and https://github.com/pytorch/examples/tree/master/dcgan

the final non linearity of the generator should be tanh ( positive and negative values, centered at zero) for GAN, but sigmoid for VAE,
where image pixel values are coded as probabilities between 0 and 1
"""


from __future__ import print_function
import torch
import torch.nn as nn

from Param import nz, nc
from Swish_Function import Swish




class Decoder256_swish(nn.Module):
    def __init__(self,nz=nz,ngf=16,nc=nc):
        super(Decoder256_swish, self).__init__()
        self.nz=nz
        self.nc=nc
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf *64 , 2, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 64),
            Swish(),
            # size ngf*64 x2 x2
            nn.ConvTranspose2d(ngf * 64, ngf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 32),
            Swish(),
            # size ngf*32 x4 x4
            nn.ConvTranspose2d(ngf*32, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            Swish(),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            Swish(),
            # state size. (ngf*4) x 16 x16
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            Swish(),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            Swish(),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            Swish(),
            # state size. (nc) x 128 x 128
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            #nn.Sigmoid() # for VAE
            # state size. (nc) x 256 x 256

        )

    def forward(self, input):
        output = self.main(input.reshape(-1, self.nz, 1, 1))
        return output


class Decoder512_swish(nn.Module):
    def __init__(self,nz=nz,ngf=8,nc=nc):
        super(Decoder512_swish, self).__init__()
        self.nz=nz
        self.nc=nc
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf *128 , 2, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 128),
            Swish(),
            # size ngf*128 x2 x2
            nn.ConvTranspose2d(ngf * 128, ngf * 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 64),
            Swish(),
            # size ngf*64 x4 x4
            nn.ConvTranspose2d(ngf * 64, ngf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 32),
            Swish(),
            # size ngf*32 x8 x8
            nn.ConvTranspose2d(ngf*32, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            Swish(),
            # state size. (ngf*16) x 16 x16
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            Swish(),
            # state size. (ngf*8) x 32 x 32
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            Swish(),
            # state size. (ngf*4) x 64 x 64
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            Swish(),
            # state size. (ngf*2) x 128 x 128
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            Swish(),
            # state size. (ngf) x 256 x 256
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            #nn.Sigmoid() # for VAE
            # state size. (nc) x 256 x 256

        )

    def forward(self, input):
        output = self.main(input.reshape(-1, self.nz, 1, 1))
        return output


