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

from Param import nz, nc, nc_2, Initializze_BG, device, method
from background_initialization import BENet

class Encoder64(nn.Module):
    def __init__(self,nz=nz,nef=64,nc=nc):
        super(Encoder64, self).__init__()
        self.nz=nz
        self.nef=nef
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef) x 32 x 32
            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*2) x 16 x 16
            nn.Conv2d(nef * 2, nef * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*4) x 8 x 8
            nn.Conv2d(nef * 4, nef * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*8) x 4 x 4
            nn.Conv2d(nef * 8, nef * 16, 4, 1, 0, bias=False),
            nn.Conv2d(nef * 16, nz, 1, 1, 0, bias=True),
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




class Encoder128(nn.Module):
    def __init__(self,nz=nz,nef=32,nc=nc):
        super(Encoder128, self).__init__()
        self.nz=nz
        self.nc=nc
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef) x 64 x 64
            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef) x 32 x 32
            nn.Conv2d(nef*2, nef * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*2) x 16 x 16
            nn.Conv2d(nef * 4, nef * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*4) x 8 x 8
            nn.Conv2d(nef * 8, nef * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*8) x 4 x 4
            nn.Conv2d(nef * 16, nef * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nef * 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nef * 32, nz, 1, 1, 0, bias=True),
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


class Encoder256(nn.Module):
    def __init__(self,nz=nz,nef=16,nc=nc):
        super(Encoder256, self).__init__()
        self.nz=nz
        self.nc=nc
        self.main = nn.Sequential(
            # input is (nc) x 258 x 256
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef) x 128 x 128
            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef) x 64 x 64
            nn.Conv2d(nef * 2, nef * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef) x 32 x 32
            nn.Conv2d(nef*4, nef * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*2) x 16 x 16
            nn.Conv2d(nef * 8, nef * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 16),
            nn.LeakyReLU(0.2, inplace=True),
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


class Encoder512(nn.Module):
    def __init__(self,nz=nz,nef=8,nc=nc):
        super(Encoder512, self).__init__()
        self.nz=nz
        self.nc=nc
        self.main = nn.Sequential(
            # input is (nc) x 512 x 512
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(0.2, inplace=True),
            # state size is (nef) x 256 x 256
            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*2) x 128 x 128
            nn.Conv2d(nef*2, nef * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*4) x 64 x 64
            nn.Conv2d(nef * 4, nef * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*8) x 32 x 32
            nn.Conv2d(nef*8, nef * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*16) x 16 x 16
            nn.Conv2d(nef * 16, nef * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*32) x 8 x 8
            nn.Conv2d(nef * 32, nef * 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*64) x 4 x 4
            nn.Conv2d(nef * 64, nef * 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nef * 128),
            nn.LeakyReLU(0.2, inplace=True),
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


class Encoder512_G(nn.Module):
    def __init__(self,nz=nz,nef=8,nc=nc):
        super(Encoder512_G, self).__init__()
        self.nz=nz
        self.nc=nc
        self.main = nn.Sequential(
            # input is (nc) x 512 x 512
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False),
            nn.GroupNorm(2, nef),
            nn.LeakyReLU(0.2, inplace=True),
            # state size is (nef) x 256 x 256
            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),
            nn.GroupNorm(nef, nef * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*2) x 128 x 128
            nn.Conv2d(nef*2, nef * 4, 4, 2, 1, bias=False),
            nn.GroupNorm(nef*2, nef * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*4) x 64 x 64
            nn.Conv2d(nef * 4, nef * 8, 4, 2, 1, bias=False),
            nn.GroupNorm(nef*4, nef * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*8) x 32 x 32
            nn.Conv2d(nef*8, nef * 16, 4, 2, 1, bias=False),
            nn.GroupNorm(nef*8, nef * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*16) x 16 x 16
            nn.Conv2d(nef * 16, nef * 32, 4, 2, 1, bias=False),
            nn.GroupNorm(nef*16, nef * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*32) x 8 x 8
            nn.Conv2d(nef * 32, nef * 64, 4, 2, 1, bias=False),
            nn.GroupNorm(nef*32, nef * 64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*64) x 4 x 4
            nn.Conv2d(nef * 64, nef * 128, 4, 1, 0, bias=False),
            nn.GroupNorm(nef*64, nef * 128),
            nn.LeakyReLU(0.2, inplace=True),
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



class Encoder1280(nn.Module):
    def __init__(self,nz=nz,nef=3,nc=nc):
        super().__init__()
        coeff = [4, 16, 64, 256, 128, 64, 16, 4, 2, 1]
        group =[1, 2, 4,  8, 1,  1, 1, 1, 1, 1, 1]
        self.nz=nz
        self.nef=nef
        self.main = nn.Sequential(
            # input is 720 x 1280
            nn.Conv2d(nc, nef*coeff[0], 4, 2, 1, bias=False, groups = group[0]),
            #  360 x 640
            nn.BatchNorm2d(nef*coeff[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nef*coeff[0], nef*coeff[1], 4, 2, 1, bias=False,groups = group[1] ),
            #  180 x 320
            nn.BatchNorm2d(nef*coeff[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nef*coeff[1], nef*coeff[2], 4, 2, 1, bias=False,groups = group[2]),
            #  90 x 160
            nn.BatchNorm2d(nef*coeff[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nef*coeff[2], nef*coeff[3], 4, 2, 1, bias=False,groups = group[3] ),
            #  45 x 80
            nn.BatchNorm2d(nef*coeff[3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nef*coeff[3], nef*coeff[4], 4, 2, 1, bias=False,groups = group[4]),
            #  22 x 40
            nn.BatchNorm2d(nef*coeff[4]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nef*coeff[4], nef*coeff[5], 4, 2, 1, bias=False, groups = group[5]),
            #  11 x 20
            nn.BatchNorm2d(nef*coeff[5]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nef*coeff[5], nef*coeff[6], 4, 2, 1, bias=False, groups = group[6]),
            #  5 x 10
            nn.BatchNorm2d(nef*coeff[6]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nef*coeff[6], nef*coeff[7], 4, 2, 1, bias=False, groups = group[7]),
            #  2 x 5
            nn.BatchNorm2d(nef*coeff[7]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nef*coeff[7], nef*coeff[8], 4, 2, 1, bias=False, groups = group[8]),
            #  1 x 2
            nn.Conv2d(nef*coeff[8], nef*coeff[9], (1,2), 1, 0, bias=False, groups = group[9]),
            #  1 x 1
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nef*coeff[9], nz, 1, 1, 0, bias=False, groups = group[10]),
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
























