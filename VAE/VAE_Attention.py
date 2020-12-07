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
from torch.autograd import Variable

from Param import nz, nc, device








class VAE_Attention_model(nn.Module):
    def __init__(self, nz=nz, nef=8, ngf=8, nc=nc):
        super(VAE_Attention_model, self).__init__()
        self.nz=nz
        self.nc=nc

        assert nef == ngf , "encoder and decoder outputs should have the same dimensions at each level"

        ###########################################
        #       Encoder's layers
        ###########################################
        self.Encoder_layer1 = nn.Sequential(
            # input is (nc) x 512 x 512
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(0.2, inplace=True))

        self.Encoder_layer2 = nn.Sequential(
            # state size is (nef) x 256 x 256
            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 2),
            nn.LeakyReLU(0.2, inplace=True))

        self.Encoder_layer3 = nn.Sequential(
            # state size. (nef*2) x 128 x 128
            nn.Conv2d(nef*2, nef * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 4),
            nn.LeakyReLU(0.2, inplace=True))

        self.Encoder_layer4 = nn.Sequential(
            # state size. (nef*4) x 64 x 64
            nn.Conv2d(nef * 4, nef * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 8),
            nn.LeakyReLU(0.2, inplace=True))

        self.Encoder_layer5 = nn.Sequential(
            # state size. (nef*8) x 32 x 32
            nn.Conv2d(nef*8, nef * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 16),
            nn.LeakyReLU(0.2, inplace=True))

        self.Encoder_layer6 = nn.Sequential(
            # state size. (nef*16) x 16 x 16
            nn.Conv2d(nef * 16, nef * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 32),
            nn.LeakyReLU(0.2, inplace=True))

        self.Encoder_layer7 = nn.Sequential(
            # state size. (nef*32) x 8 x 8
            nn.Conv2d(nef * 32, nef * 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 64),
            nn.LeakyReLU(0.2, inplace=True))

        self.Encoder_layer8 = nn.Sequential(
            # state size. (nef*64) x 4 x 4
            nn.Conv2d(nef * 64, nef * 128, 3, 1, 0, bias=False),
            nn.BatchNorm2d(nef * 128),
            nn.LeakyReLU(0.2, inplace=True))

        self.Encoder_layer9 = nn.Sequential(
            # state size. (nef*128) x 2 x 2
            nn.Conv2d(nef * 128, nz, 2, 1, 0, bias=True),
            nn.Sigmoid()
            # state size. (nz) x 1 x 1
        )

        ###########################################
        #       Decoder's layers
        ###########################################
            # state size. (nz) x 1 x 1
        self.Decoder_layer1 = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf *128 , 2, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 128),
            nn.ReLU(True))

        self.Decoder_layer2 = nn.Sequential(
            # size ngf*128 x2 x2
            nn.ConvTranspose2d(ngf * 128, ngf * 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 64),
            nn.ReLU(True))

        self.Decoder_layer3 = nn.Sequential(
            # size ngf*64 x4 x4
            nn.ConvTranspose2d(ngf * 64, ngf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True))

        self.Decoder_layer4 = nn.Sequential(
            # size ngf*32 x8 x8
            nn.ConvTranspose2d(ngf*32, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True))

        self.Decoder_layer5 = nn.Sequential(
            # state size. (ngf*16) x 16 x16
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True))

        self.Decoder_layer6 = nn.Sequential(
            # state size. (ngf*8) x 32 x 32
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True))

        self.Decoder_layer7 = nn.Sequential(
            # state size. (ngf*4) x 64 x 64
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True))

        self.Decoder_layer8 = nn.Sequential(
            # state size. (ngf*2) x 128 x 128
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True))

        self.Decoder_layer9 = nn.Sequential(
            # state size. (ngf) x 256 x 256
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            # nn.Tanh()
            nn.Sigmoid() # for VAE
            # state size. (nc) x 512 x 512
        )


        #------------------------
        #    Attention Modules 
        #------------------------
        self.attention_layer = Attention_layer(nef)
        self.attention_layer2 = Attention_layer(nef*2)
        self.attention_layer4 = Attention_layer(nef*4)
        self.attention_layer8 = Attention_layer(nef*8)
        self.attention_layer16 = Attention_layer(nef*16)
        self.attention_layer32 = Attention_layer(nef*32)
        self.attention_layer64 = Attention_layer(nef*64)
        self.attention_layer128 = Attention_layer(nef*128)


        #
        self.fc1 = nn.Linear(nz, 64)
        self.fc2 = nn.Linear(nz, 64)
        self.fc3 = nn.Linear(64, nz)


    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()

        eps = torch.FloatTensor(std.size()).normal_().to(device)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, input):
        batch_size = input.shape[0]
        num_channels = input.shape[1]

        input_img = input[:,:3,:,:]
        input_flow = input[:,3:num_channels,:,:]


        #########################
        #      Encoding
        #########################
        # N x nc x 512 x 512
        x1 = self.Encoder_layer1(input_img)
        m_x1 = self.Encoder_layer1(input_flow)

        # N x nef x 256 x 256
        x2 = self.Encoder_layer2(x1)
        m_x2 = self.Encoder_layer2(m_x1)

        # N x nef*2 x 128 x 128
        x3 = self.Encoder_layer3(x2)
        m_x3 = self.Encoder_layer3(m_x2)

        # N x nef*4 x 64 x 64
        x4 = self.Encoder_layer4(x3)
        m_x4 = self.Encoder_layer4(m_x3)

        # N x nef*8 x 32 x 32
        x5 = self.Encoder_layer5(x4)
        m_x5 = self.Encoder_layer5(m_x4)

        # N x nef*16 x 16 x 16
        x6 = self.Encoder_layer6(x5)
        m_x6 = self.Encoder_layer6(m_x5)

        # N x nef*32 x 8 x 8
        x7 = self.Encoder_layer7(x6)
        m_x7 = self.Encoder_layer7(m_x6)

        # N x nef*64 x 4 x 4
        x8 = self.Encoder_layer8(x7)
        m_x8 = self.Encoder_layer8(m_x7)

        # N x nef*128 x 2 x 2
        Encoder_out = self.Encoder_layer9(x8)
        m_Encoder_out = self.Encoder_layer9(m_x8)

        # N x nz x 1 x 1
        Encoder_out = Encoder_out.reshape(batch_size, self.nz, 1, 1)
        m_Encoder_out = m_Encoder_out.reshape(batch_size, self.nz, 1, 1)


        Encoder_out = Encoder_out.view(batch_size, self.nz)
        m_Encoder_out = m_Encoder_out.view(batch_size, self.nz)

        mu = self.fc1(Encoder_out)   #fc1
        logvar = self.fc2(Encoder_out)  #fc2
        z = self.reparametrize(mu, logvar)
        z = self.fc3(z).reshape(-1, self.nz, 1, 1)  #fc3

        m_mu = self.fc1(m_Encoder_out)   #fc1
        m_logvar = self.fc2(m_Encoder_out)  #fc2
        m_z = self.reparametrize(m_mu, m_logvar)
        m_z = self.fc3(m_z).reshape(-1, self.nz, 1, 1)  #fc3

        #########################
        #      Decoding
        #########################
        # N x nz x 1 x 1
        y1 = self.Decoder_layer1(z)
        m_y1 = self.Decoder_layer1(m_z)
        #print(m_x8.shape, m_y1.shape)
        weight1 = self.attention_layer128(m_x8, m_y1)

        # N x ngf*128 x 2 x 2
        a1 = torch.mul(x8, weight1) + y1
        y2 = self.Decoder_layer2(a1)
        m_y2 = self.Decoder_layer2(m_y1)
        weight2 = self.attention_layer64(m_x7, m_y2)

        del y1, m_y1, a1, weight1, m_x7, m_x8, x8
        
        # N x ngf*64 x 4 x 4
        a2 = torch.mul(x7, weight2) + y2
        y3 = self.Decoder_layer3(a2)
        m_y3 = self.Decoder_layer3(m_y2)
        weight3 = self.attention_layer32(m_x6, m_y3)

        del y2, m_y2, a2, weight2, m_x6, x7

        # N x ngf*32 x 8 x 8
        a3 = torch.mul(x6, weight3) + y3
        y4 = self.Decoder_layer4(a3)
        m_y4 = self.Decoder_layer4(m_y3)
        weight4 = self.attention_layer16(m_x5,m_y4)

        del y3, m_y3, a3, weight3, m_x5, x6

        # N x ngf*16 x 16 x 16
        a4 = torch.mul(x5, weight4) + y4
        y5 = self.Decoder_layer5(a4)
        m_y5 = self.Decoder_layer5(m_y4)
        weight5 = self.attention_layer8(m_x4,m_y5)

        del y4, m_y4, a4, weight4, m_x4, x5

        # N x ngf*8 x 32 x 32
        a5 = torch.mul(x4, weight5) + y5
        y6 = self.Decoder_layer6(a5)
        m_y6 = self.Decoder_layer6(m_y5)
        weight6 = self.attention_layer4(m_x3,m_y6)

        del y5, m_y5, a5, weight5, m_x3, x4

        # N x ngf*4 x 64 x 64
        a6 = torch.mul(x3, weight6) + y6
        y7 = self.Decoder_layer7(a6)
        m_y7 = self.Decoder_layer7(m_y6)
        weight7 = self.attention_layer2(m_x2,m_y7)

        del y6, m_y6, a6, weight6, m_x2, x3

        # N x ngf*2 x 128 x 128
        a7 = torch.mul(x2, weight7) + y7
        y8 = self.Decoder_layer8(a7)
        m_y8 = self.Decoder_layer8(m_y7)
        weight8 = self.attention_layer(m_x1,m_y8)

        del y7, m_y7, a7, weight7, m_x1, x2

        # N x ngf x 256 x 256
        a8 = torch.mul(x1, weight8) + y8
        out = self.Decoder_layer9(a8)
        m_out = self.Decoder_layer9(m_y8)

        del y8, m_y8, a8, weight8, x1

        torch.cuda.empty_cache()

        return out.reshape(batch_size, self.nc, 512, 512), mu, logvar



def attention_logit(f):
    return nn.Conv2d(f, f, 1, 1, 0, bias=True), nn.Sigmoid() # nn.Softmax() #

class Attention_layer(nn.Module):
    def __init__(self, n):
        self.n = n
        super().__init__()

        self.attention_logit = nn.Sequential(*attention_logit(self.n))
                                

    def forward(self, Enc, Dec):
        b1, n1, h1, w1 = Enc.shape
        b2, n2, h2, w2 = Dec.shape

        assert b1 == b2 and n1 == n2 and h1 == h2 and w1 == w2, "encoder and decoder outputs should have the same dimensions"
        h, w, n = h1, w1, n1
        batch_size = b1

        y = self.attention_logit(Enc) # y dimension: n x h x w
        out = torch.ones(y.shape).to(device) - y

        #x = torch.mul(Dec, y)  # pixel-wise multiplication

        #out = x + Dec

        return out.view(batch_size, n, h, w)


class Renet(nn.Module):
    def __init__(self, size, in_channel, out_channel):
        super(Renet, self).__init__()
        self.size = size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.vertical = nn.LSTM(input_size=in_channel, hidden_size=256, batch_first=True,
                                bidirectional=True)  # each row
        self.horizontal = nn.LSTM(input_size=512, hidden_size=256, batch_first=True,
                                  bidirectional=True)  # each column
        self.conv = nn.Conv2d(512, out_channel, 1)

    def forward(self, *input):
        x = input[0]
        temp = []
        x = torch.transpose(x, 1, 3)  # batch, width, height, in_channel
        for i in range(self.size):
            h, _ = self.vertical(x[:, :, i, :])
            temp.append(h)  # batch, width, 512
        x = torch.stack(temp, dim=2)  # batch, width, height, 512
        temp = []
        for i in range(self.size):
            h, _ = self.horizontal(x[:, i, :, :])
            temp.append(h)  # batch, width, 512
        x = torch.stack(temp, dim=3)  # batch, height, 512, width
        x = torch.transpose(x, 1, 2)  # batch, 512, height, width
        x = self.conv(x)
        return x





