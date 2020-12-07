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

from Swish_Function import Swish

from Param import nz, nc, nc_2, Initializze_BG, device, method
from background_initialization import BENet


class Encoder256_swish(nn.Module):
    def __init__(self,nz=nz,nef=16,nc=nc):
        super(Encoder256_swish, self).__init__()
        self.nz=nz
        self.nc=nc
        self.main = nn.Sequential(
            # input is (nc) x 258 x 256
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef),
            Swish(),
            # state size. (nef) x 128 x 128
            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 2),
            Swish(),
            # state size. (nef) x 64 x 64
            nn.Conv2d(nef * 2, nef * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 4),
            Swish(),
            # state size. (nef) x 32 x 32
            nn.Conv2d(nef*4, nef * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 8),
            Swish(),
            # state size. (nef*2) x 16 x 16
            nn.Conv2d(nef * 8, nef * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 16),
            Swish(),
            # state size. (nef*4) x 8 x 8
            nn.Conv2d(nef * 16, nef * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*8) x 4 x 4
            nn.Conv2d(nef * 32, nef * 64, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nef * 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nef * 64, nz, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input, Initializze_BG=Initializze_BG):
        if Initializze_BG:        
            if method == 'Combination': BEN_model = BENet(nc=nc+nc_2).to(device)
            else: BEN_model = BENet().to(device)
            background, foreground = BEN_model(input)
            output = self.main(foreground)
            return output.reshape(-1, self.nz), background

        output = self.main(input)
        return output.reshape(-1, self.nz)


class Encoder512_swish(nn.Module):
    def __init__(self,nz=nz,nef=8,nc=nc):
        super(Encoder512_swish, self).__init__()
        self.nz=nz
        self.nc=nc
        self.main = nn.Sequential(
            # input is (nc) x 512 x 512
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef),
            Swish(),
            # state size is (nef) x 256 x 256
            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 2),
            Swish(),
            # state size. (nef*2) x 128 x 128
            nn.Conv2d(nef*2, nef * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 4),
            Swish(),
            # state size. (nef*4) x 64 x 64
            nn.Conv2d(nef * 4, nef * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 8),
            Swish(),
            # state size. (nef*8) x 32 x 32
            nn.Conv2d(nef*8, nef * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 16),
            Swish(),
            # state size. (nef*16) x 16 x 16
            nn.Conv2d(nef * 16, nef * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 32),
            Swish(),
            # state size. (nef*32) x 8 x 8
            nn.Conv2d(nef * 32, nef * 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 64),
            Swish(),
            # state size. (nef*64) x 4 x 4
            nn.Conv2d(nef * 64, nef * 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nef * 128),
            Swish(),
            nn.Conv2d(nef * 128, nz, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input, Initializze_BG=Initializze_BG):
        if Initializze_BG:        
            if method == 'Combination': BEN_model = BENet(nc=nc+nc_2).to(device)
            else: BEN_model = BENet().to(device)
            background, foreground = BEN_model(input)
            output = self.main(foreground)
            return output.reshape(-1, self.nz), background

        output = self.main(input)
        return output.reshape(-1, self.nz)




