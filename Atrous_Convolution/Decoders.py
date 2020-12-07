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


class Decoder64(nn.Module):

    def __init__(self,nz=nz,ngf=64,nc=nc):
        super(Decoder64, self).__init__()
        self.nz=nz
        self.nc=nc
        self.main = nn.Sequential(
            # input is z, going into a convolution
            # input shape bachsize x nz
            nn.Conv2d(nz, ngf * 16, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()  # for GAN
            #nn.Sigmoid() # for VAE
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input.reshape(-1, self.nz, 1, 1))
        return output


class Decoder128(nn.Module):
    def __init__(self,nz=nz,ngf=32,nc=nc):
        super(Decoder128, self).__init__()
        self.nz = nz
        self.nc = nc
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf *32 , 2, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True),
            # size ngf*32 x2 x2
            nn.ConvTranspose2d(ngf*32, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # size ngf*16 x4 x4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh() # for GAN
            #nn.Sigmoid() # for VAE
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        output = self.main(input.reshape(-1, self.nz, 1, 1))
        return output



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


class Decoder512(nn.Module):
    def __init__(self,nz=nz,ngf=8,nc=nc):
        super(Decoder512, self).__init__()
        self.nz=nz
        self.nc=nc
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf *128 , 2, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 128),
            nn.ReLU(True),
            # size ngf*128 x2 x2
            nn.ConvTranspose2d(ngf * 128, ngf * 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 64),
            nn.ReLU(True),
            # size ngf*64 x4 x4
            nn.ConvTranspose2d(ngf * 64, ngf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True),
            # size ngf*32 x8 x8
            nn.ConvTranspose2d(ngf*32, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 16 x16
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 32 x 32
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 64 x 64
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 128 x 128
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 256 x 256
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            #nn.Sigmoid() # for VAE
            # state size. (nc) x 256 x 256

        )

    def forward(self, input):
        output = self.main(input.reshape(-1, self.nz, 1, 1))
        return output




class Decoder1280(nn.Module):

    def __init__(self,nz=nz,ngf=3,nc=nc):
        super().__init__()
        self.nz=nz
        self.nc=nc
        coeff = [4, 16, 64, 256, 128, 64, 16, 4, 2, 1]
        group =[1, 2, 4,  8,  1,  1,  1, 1, 1, 1, 1]
        self.main = nn.Sequential(
            nn.Conv2d(nz, ngf*coeff[9] , 1, 1, 0, bias=False, groups =group[10]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf*coeff[9], ngf*coeff[8], (1,2), 1, 0, bias=False, groups =group[9]),
            # image size 1 x 2
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf*coeff[8], ngf*coeff[7], 4, 2, 1,output_padding=(0,1), bias=False, groups =group[8]),
            #  2 x 5
            nn.BatchNorm2d(ngf*coeff[7]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf*coeff[7], ngf*coeff[6], 4, 2, 1, output_padding=(1, 0), bias=False, groups =group[7]),
            #  5 x 10
            nn.BatchNorm2d(ngf*coeff[6]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf*coeff[6], ngf*coeff[5], 4, 2, 1, output_padding=(1, 0), bias=False, groups =group[6]),
            #  11 x 20
            nn.BatchNorm2d(ngf*coeff[5]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf*coeff[5], ngf*coeff[4], 4, 2, 1, bias=False, groups =group[5]),
            #  22 x 40
            nn.BatchNorm2d(ngf*coeff[4]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf*coeff[4], ngf*coeff[3], 4, 2, 1, output_padding=(1, 0), bias=False, groups =group[4]),
            #  45 x 80
            nn.BatchNorm2d(ngf*coeff[3] ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf*coeff[3], ngf*coeff[2], 4, 2, 1,  bias=False, groups =group[3]),
            # 90 x 160
            nn.BatchNorm2d(ngf*coeff[2] ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf*coeff[2], ngf*coeff[1], 4, 2, 1,  bias=False, groups =group[2]),
            #  180 x 320
            nn.BatchNorm2d(ngf*coeff[1] ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf*coeff[1], ngf*coeff[0], 4, 2, 1,  bias=False, groups =group[1]),
            #  360 x 640
            nn.BatchNorm2d(ngf*coeff[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf*coeff[0], nc, 4, 2, 1, bias=True, groups =group[0]),
            nn.Tanh()
            # 720 x 1280
        )
    def forward(self, input):
        output = self.main(input.reshape(-1, self.nz, 1, 1))
        return output


