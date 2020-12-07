from __future__ import print_function
import torch
import torch.nn as nn
import torchvision.models as models

from Param import nz, nc

class Encoder256(nn.Module):
    def __init__(self, model, nc=nc, nz=nz):
        super(Encoder256, self).__init__()

        self.nc = nc
        self.nz = nz

        self.model = model

        if self.model == 'alexnet':
            model_pre = models.alexnet(pretrained=True)
            for p in model_pre.parameters():
                p.requires_grad = False

            self.features_map = model_pre.features
            
            self.Conv_layers = nn.Sequential(
                               # batch x 256 x 7 x 7
                               nn.Conv2d(256, 512, 3, 2, 1, bias=False),
                               nn.BatchNorm2d(512),
                               nn.LeakyReLU(0.2, inplace=True),
                               # batch x 512 x 4 x 4
                               nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
                               nn.BatchNorm2d(1024),
                               nn.LeakyReLU(0.2, inplace=True),
                               # batch x 1024 x 2 x 2
                               nn.Conv2d(1024, 2048, 2, 1, 0, bias=False)
                               # batch x 2048 x 1 x 1
                                   )
       
        elif self.model == 'resnet50':
            model_pre = models.resnet50(pretrained=True)
            for p in model_pre.parameters():
                p.requires_grad = False
            layers = []
            for child in model_pre.children():
                layers.append(child)

            del layers[-1]

            self.features_map = nn.ModuleList(layers)
            self.conv_layer = nn.Conv2d(2048, 2048, 2, 1, 0, bias=False)

        else:
            assert 1==1, "Undefined model..."

        self.linear1 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, nz)

    def forward(self, input, verbose=False):

        if self.model == 'alexnet':
            x = self.features_map(input)
            if verbose: print(x.shape)

            x = self.Conv_layers(x)
            if verbose: print(x.shape)

        elif self.model == 'resnet50':
            x = input
            for i in range(len(self.features_map)):
                x = self.features_map[i](x)
                
            x = self.conv_layer(x)

        if verbose: print(x.shape)
        x = x.view(-1, 2048)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.linear3(x)
        if verbose: print(x.shape)

        return x.view(-1, self.nz, 1, 1)


class Encoder512(nn.Module):
    def __init__(self, model, nc=nc, nz=nz):
        super(Encoder512, self).__init__()

        self.nc = nc
        self.nz = nz
        self. model = model

        if self.model == 'alexnet':
            model_pre = models.alexnet(pretrained=True)
            for p in model_pre.parameters():
                p.requires_grad = False
            self.features_map = model_pre.features

            self.Conv_layers = nn.Sequential(
                               # batch x 256 x 15 x 15
                               nn.Conv2d(256, 512, 3, 2, 1, bias=False),
                               nn.BatchNorm2d(512),
                               nn.LeakyReLU(0.2, inplace=True),
                               # batch x 512 x 8 x 8
                               nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
                               nn.BatchNorm2d(1024),
                               nn.LeakyReLU(0.2, inplace=True),
                               # batch x 1024 x 4 x 4
                               nn.Conv2d(1024, 2048, 4, 2, 1, bias=False),
                               nn.BatchNorm2d(2048),
                               nn.LeakyReLU(0.2, inplace=True),
                               # batch x 2048 x 2 x 2
                               nn.Conv2d(2048, 2048, 2, 1, 0, bias=False)
                               # batch x 2048 x 1 x 1
                                   )


        elif self.model == 'resnet50':
            model_pre = models.resnet50(pretrained=True)
            for p in model_pre.parameters():
                p.requires_grad = False
            layers = []
            for child in model_pre.children():
                layers.append(child)

            del layers[-1]

            self.features_map = nn.ModuleList(layers)
            self.conv_layer = nn.Conv2d(2048, 2048, 2, 1, 0, bias=False)

        else:
            assert 1==1, 'Undefined model.'

        self.linear1 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, nz)

    def forward(self, input, verbose=False):

        if self.model == 'alexnet':
            x = self.features_map(input)
            if verbose: print(x.shape)

            x = self.Conv_layers(x)
            if verbose: print(x.shape)

        elif self.model == 'resnet50':
            x = input
            for i in range(len(self.features_map)):
                x = self.features_map[i](x)
            x = self.conv_layer(x)

        x = x.view(-1, 2048)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.linear3(x)
        if verbose: print(x.shape)

        return x.view(-1, self.nz, 1, 1)


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


