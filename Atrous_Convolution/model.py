import torch
import torch.nn as nn

#from Param import nz, nc
nz=100
nc=3


"""
class Model256(nn.Module):
    def __init__(self,nz=nz,nef=16,nc=nc):
        super(Model256, self).__init__()
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
            # nn.Tanh()
            nn.Sigmoid() # for VAE
            # state size. (nc) x 256 x 256

        )

    def forward(self, input):
        output = self.decode(input.reshape(-1, self.nz, 1, 1))
        return output
"""



class Model512(nn.Module):
    def __init__(self,nz=nz,nef=8,ngf=8,nc=nc):
        super(Model512, self).__init__()
        self.nz=nz
        self.nc=nc

        self.atrous_encoding_layer1 = nn.Conv2d(nc, nef, 4, stride=16, padding=0, dilation=5, 
                                                                                  bias=False)
        self.atrous_encoding_layer2 = nn.Conv2d(nef, nef, 4, stride=8, padding=1, dilation=3, 
                                                                                  bias=False)
        self.atrous_encoding_layer3 = nn.Conv2d(nef*2, nef*4, 4, stride=4, padding=0, dilation=1, 
                                                                                  bias=False)

        self.AvgPool = nn.AvgPool2d(4, stride=4, padding=0)

        self.encoding_layer1 = nn.Sequential(
            # input is (nc) x 512 x 512
            nn.Conv2d(nc, nef, 4, 2, 1, bias=True),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(0.2, inplace=True))

        self.encoding_layer2 = nn.Sequential(
            # state size is (nef) x 256 x 256
            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 2),
            nn.LeakyReLU(0.2, inplace=True))

        self.encoding_layer3 = nn.Sequential(
            # state size. (nef*2) x 128 x 128
            nn.Conv2d(nef*2, nef * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*4) x 64 x 64
            nn.Conv2d(nef * 4, nef * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 8),
            nn.LeakyReLU(0.2, inplace=True))

        self.encoding_layer4 = nn.Sequential(
            # state size. (nef*16) x 32 x 32
            nn.Conv2d(nef*16, nef * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*16) x 16 x 16
            nn.Conv2d(nef * 32, nef * 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*32) x 8 x 8
            nn.Conv2d(nef * 64, nef * 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*64) x 4 x 4
            nn.Conv2d(nef * 128, nef * 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nef * 256),
            nn.LeakyReLU(0.2, inplace=True))

        self.fc1 = nn.Linear(nef*256, nef*64)
        self.dropout = nn.Dropout(p=0.2, inplace=True)
        self.fc2 = nn.Linear(nef*64, nz)
        self.encode_out = nn.Sigmoid()
        self.fc3 = nn.Linear(nz, nef*64)
        self.fc4 = nn.Linear(nef*64, nef*256)

        self.decoding_layer1 = nn.Sequential(
            nn.ConvTranspose2d(nef*256, ngf *128 , 2, 1, 0, bias=False),
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
            nn.ReLU(True))

        self.decoding_layer2 = nn.Sequential(
            # state size. (ngf*16) x 32 x 32
            nn.ConvTranspose2d(ngf * 24, ngf * 4, 4, 2, 1, bias=False),
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
            # state size. (nc) x 512 x 512

        )

    def forward(self, input, verbose=False):
        b_size = input.shape[0]
        if verbose: print('input: ', input.shape)

        x1 = self.encoding_layer1(input)
        x2 = self.encoding_layer2(x1)
        x3 = self.encoding_layer3(x2)

        if verbose: print('Enc1: ', x1.shape)
        if verbose: print('Enc2: ', x2.shape)
        if verbose: print('Enc3: ', x3.shape)

        a1 = self.atrous_encoding_layer1(input) # nef x 32 x 32
        a2 = self.atrous_encoding_layer2(x1) # nef x 32 x 32
        a3 = self.atrous_encoding_layer3(x2) # nef*4 x 32 x 32
        a4 = self.AvgPool(x2) # nef*2 x 32 x 32

        if verbose: print('a1: ', a1.shape)
        if verbose: print('a2: ', a2.shape)
        if verbose: print('a3: ', a3.shape)
        if verbose: print('a4: ', a4.shape)

        x = torch.cat((x3, a1, a2, a3, a4),1)  # nef*16 x 32 x 32
        if verbose: print('Enc4: ', x.shape)

        x4 = self.encoding_layer4(x) # nef*256 x 1 x 1
        x4 = x4.view(b_size, x4.shape[1]*x4.shape[2]*x4.shape[3]) 
        x5 = self.dropout(self.fc1(x4))
        #x6 = self.encode_out(self.fc2(x5))
        x6 = self.fc2(x5)
        x7 = self.fc3(x6)
        x8 = self.fc4(x7)
        x8 = x8.view(b_size, x8.shape[1], 1, 1)

        if verbose: print('Enc5: ', x4.shape)
        if verbose: print('linear1: ', x5.shape)
        if verbose: print('linear2: ', x6.shape)
        if verbose: print('linear3: ', x7.shape)
        if verbose: print('linear4: ', x8.shape)

        x9 = self.decoding_layer1(x8)
        x = torch.cat((x9, x),1)  # nef*16 x 32 x 32
        output = self.decoding_layer2(x)

        if verbose: print('Dec1:', x9.shape)
        if verbose: print('Dec2:', x.shape)
        if verbose: print('Dec3:', output.shape)
 
        return output.view(b_size, nc, 512, 512)




