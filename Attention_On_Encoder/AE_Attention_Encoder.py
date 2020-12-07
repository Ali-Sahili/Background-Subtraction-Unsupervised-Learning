from __future__ import print_function
import torch
import torch.nn as nn

from Attention_On_Encoder.Attention_logits import ChannelwiseAttention, SpatialAttention, CBAM_Module

from Param import nz, nc, ATTENTION_M


class AE_Attention_Encoder(nn.Module):
    def __init__(self, nz=nz, nef=8, ngf=8, nc=nc):
        super(AE_Attention_Encoder, self).__init__()
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
        self.Decode = nn.Sequential(
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
            # state size. (nc) x 512 x 512
        )


        #------------------------
        #    Attention Modules 
        #------------------------
        if ATTENTION_M == 'Spatio_ChannelWise':
            self.attention_layer1 = SpatialAttention(nef)
            self.attention_layer2 = SpatialAttention(nef*2)
            self.attention_layer3 = SpatialAttention(nef*4)
            self.attention_layer4 = SpatialAttention(nef*8)
            self.attention_layer5 = ChannelwiseAttention(nef*16)
            self.attention_layer6 = ChannelwiseAttention(nef*32)
            self.attention_layer7 = ChannelwiseAttention(nef*64)
            self.attention_layer8 = ChannelwiseAttention(nef*128)

        elif ATTENTION_M == 'CBAM_Module':
            self.reduction = 2
            self.attention_layer1 = CBAM_Module(nef, self.reduction)
            self.attention_layer2 = CBAM_Module(nef*2, self.reduction)
            self.attention_layer3 = CBAM_Module(nef*4, self.reduction)
            self.attention_layer4 = CBAM_Module(nef*8, self.reduction)
            self.attention_layer5 = CBAM_Module(nef*16, self.reduction)
            self.attention_layer6 = CBAM_Module(nef*32, self.reduction)
            self.attention_layer7 = CBAM_Module(nef*64, self.reduction)
            self.attention_layer8 = CBAM_Module(nef*128, self.reduction)

        else:
            assert 1==1, "Undefined attention modules..."

    def forward(self, input, verbose=False):

        #########################
        #      Encoding
        #########################
        # N x nc x 512 x 512
        x1 = self.Encoder_layer1(input)
        if verbose: print('shape  of x1: ', x1.shape)

        # N x nef x 256 x 256
        m_x1 = self.attention_layer1(x1)
        a1 = x1 * m_x1
        x2 = self.Encoder_layer2(a1)

        if verbose: print('shape  of x2: ', x2.shape)
        if verbose: print('shape  of attention map 1: ', a1.shape)

        # N x nef*2 x 128 x 128
        m_x2 = self.attention_layer2(x2)
        a2 = x2 * m_x2
        x3 = self.Encoder_layer3(a2)

        if verbose: print('shape  of x3: ', x3.shape)
        if verbose: print('shape  of attention map 2: ', a2.shape)

        # N x nef*4 x 64 x 64
        m_x3 = self.attention_layer3(x3)
        a3 = x3 * m_x3
        x4 = self.Encoder_layer4(a3)

        if verbose: print('shape  of x4: ', x4.shape)
        if verbose: print('shape  of attention map 3: ', a3.shape)

        # N x nef*8 x 32 x 32
        m_x4 = self.attention_layer4(x4)
        a4 = x4 * m_x4
        x5 = self.Encoder_layer5(a4)

        if verbose: print('shape  of x5: ', x5.shape)
        if verbose: print('shape  of attention map 4: ', a4.shape)

        # N x nef*16 x 16 x 16
        m_x5 = self.attention_layer5(x5)
        a5 = x5 * m_x5
        x6 = self.Encoder_layer6(a5)

        if verbose: print('shape  of x6: ', x6.shape)
        if verbose: print('shape  of attention map 5: ', a5.shape)

        # N x nef*32 x 8 x 8
        m_x6 = self.attention_layer6(x6)
        a6 = x6 * m_x6
        x7 = self.Encoder_layer7(a6)

        if verbose: print('shape  of x7: ', x7.shape)
        if verbose: print('shape  of attention map 6: ', a6.shape)

        # N x nef*64 x 4 x 4
        m_x7 = self.attention_layer7(x7)
        a7 = x7 * m_x7
        x8 = self.Encoder_layer8(a7)

        if verbose: print('shape  of x8: ', x8.shape)
        if verbose: print('shape  of attention map 7: ', a7.shape)

        # N x nef*128 x 2 x 2
        m_x8 = self.attention_layer8(x8)
        a8 = x8 * m_x8
        Encoder_out = self.Encoder_layer9(a8)

        if verbose: print('Encoder output shape: ', Encoder_out.shape)
        if verbose: print('shape  of attention map 8: ', a8.shape)
     
        #########################
        #      Decoding
        #########################
        # N x nz x 1 x 1
        out = self.Decode(Encoder_out)
        if verbose: print('Decoder output shape: ', out.shape)

        return out


