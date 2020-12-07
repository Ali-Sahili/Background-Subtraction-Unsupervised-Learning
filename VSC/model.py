import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from Param import *



class VSC512(nn.Module):
    def __init__(self,nz=nz,ngf=8,nef=8,nc=3):
        super(VSC512, self).__init__()
        self.nz=nz
        self.nc=nc

        self.encode = nn.Sequential(
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


        self.decode = nn.Sequential(
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
        )

        self.fc1 = nn.Linear(nz, 64)
        self.fc2 = nn.Linear(nz, 64)
        self.fc_logspike = nn.linear(nz, 64)
        self.fc3 = nn.Linear(64, nz)
        self.c = 50.0

    def reparameterize(self, mu, logvar, logspike):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        gaussian = eps.mul(std).add_(mu)
        eta = torch.rand_like(std)
        selection = nn.Sigmoid()(self.c*(eta + logspike.exp() - 1))
        return selection.mul(gaussian)


    def forward(self, input):
        b_size = input.shape[0]
        x = self.encode(input).view(b_size, nz)
        mu = self.fc1(x)   #fc1
        logvar = self.fc2(x)  #fc2
        logspike = -F.relu(-self.fc_logspike(x))
        z = self.reparameterize(mu, logvar, logspike)
        z = self.fc3(z).reshape(-1, self.nz, 1, 1)  #fc3
        #del  x
        return self.decode(z), mu, logvar, logspike




class VSC256(nn.Module):
    def __init__(self,nz=nz,ngf=16,nef=16,nc=3):
        super(VSC256, self).__init__()
        self.nz=nz
        self.nc=nc

        self.encode = nn.Sequential(
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


        self.decode = nn.Sequential(
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

        self.fc1 = nn.Linear(nz, 64)
        self.fc2 = nn.Linear(nz, 64)
        self.fc_logspike = nn.Linear(nz, 64)
        self.fc3 = nn.Linear(64, nz)
        self.c = 50.0

    def reparameterize(self, mu, logvar, logspike):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        gaussian = eps.mul(std).add_(mu)
        eta = torch.rand_like(std)
        selection = nn.Sigmoid()(self.c*(eta + logspike.exp() - 1))
        return selection.mul(gaussian)


    def forward(self, input):
        b_size = input.shape[0]
        x = self.encode(input).view(b_size, nz)
        mu = self.fc1(x)   #fc1
        logvar = self.fc2(x)  #fc2
        logspike = -F.relu(-self.fc_logspike(x))
        z = self.reparameterize(mu, logvar, logspike)
        z = self.fc3(z).reshape(-1, self.nz, 1, 1)  #fc3
        #del  x
        return self.decode(z), mu, logvar, logspike





class VSC128(nn.Module):
    def __init__(self,nz=nz,ngf=32,nef=32,nc=3):
        super(VSC128, self).__init__()
        self.nz=nz
        self.nc=nc

        self.encode = nn.Sequential(
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


        self.decode = nn.Sequential(
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

        self.fc1 = nn.Linear(nz, 64)
        self.fc2 = nn.Linear(nz, 64)
        self.fc_logspike = nn.linear(nz, 64)
        self.fc3 = nn.Linear(64, nz)
        self.c = 50.0

    def reparameterize(self, mu, logvar, logspike):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        gaussian = eps.mul(std).add_(mu)
        eta = torch.rand_like(std)
        selection = nn.Sigmoid()(self.c*(eta + logspike.exp() - 1))
        return selection.mul(gaussian)


    def forward(self, input):
        b_size = input.shape[0]
        x = self.encode(input).view(b_size, nz)
        mu = self.fc1(x)   #fc1
        logvar = self.fc2(x)  #fc2
        logspike = -F.relu(-self.fc_logspike(x))
        z = self.reparameterize(mu, logvar, logspike)
        z = self.fc3(z).reshape(-1, self.nz, 1, 1)  #fc3
        #del  x
        return self.decode(z), mu, logvar, logspike




class VSC_Encoder(nn.Module):
    def __init__(self,nz=nz,nef=8,nc=3):
        super(VSC_Encoder, self).__init__()
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


        self.fc1 = nn.Linear(nz, 1024)
        self.fc2 = nn.Linear(nz, 1024)
        self.fc_logspike = nn.Linear(nz, 1024)


    def forward(self, x):
        b_size = x.shape[0]
        x = self.main(x).view(b_size, nz)
        mu = self.fc1(x)
        logvar = self.fc2(x)
        logspike = -F.relu(-self.fc_logspike(x))

        return mu, logvar, logspike



class VSC_Decoder(nn.Module):
    def __init__(self,nz=nz,ngf=8,nc=3):
        super(VSC_Decoder, self).__init__()
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
        )

        self.fc = nn.Linear(1024, nz)
        self.c = 50.0

    def reparameterize(self, mu, logvar, logspike):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        gaussian = eps.mul(std).add_(mu)
        eta = torch.rand_like(std)
        selection = nn.Sigmoid()(self.c*(eta + logspike.exp() - 1))
        return selection.mul(gaussian)


    def forward(self, mu, logvar, logspike):
        z = self.reparameterize(mu, logvar, logspike)
        z = z.view(-1, 1024)
        z = self.fc(z).reshape(-1, self.nz, 1, 1)

        return self.main(z)




