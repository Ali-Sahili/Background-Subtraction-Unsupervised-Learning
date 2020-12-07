from __future__ import print_function
import torch
import torch.nn as nn

from Param import nz, nc

# encoder and decoder for 512x512 image using 64x64 patches

class Encoder_Layer_F(nn.Module):
    def __init__(self, nc, nz, h, w, hlatent, wlatent, unfold_kernel, unfold_stride, conv_stride, 
                    conv_padding, conv_kernel):
        super().__init__()
        self.nz = nz  # number of channels of latents to produce
        self.nc = nc  # number of channels of input image
        self.h = h  # input image height
        self.w = w  #

        assert h % conv_stride == 0
        assert w % conv_stride == 0
        assert hlatent == h // conv_stride
        assert wlatent == w // conv_stride

        self.hlatent = hlatent  # latent image height
        self.wlatent = wlatent
        self.unfold_kernel = unfold_kernel  # size of patch
        self.unfold_stride = unfold_stride  # stride of patch ( could be different from patche size)

        assert unfold_kernel % conv_stride == 0

        self.conv_stride = conv_stride  # stride of the convolutions

        assert h % unfold_stride == 0
        assert w % unfold_stride == 0

        self.H = h // unfold_stride  # H*W = number of patchs
        self.W = w // unfold_stride

        self.unfold = nn.Unfold(unfold_kernel, stride=unfold_stride)
        self.conv = nn.Sequential(
            nn.Conv2d(self.H * self.W * nc, self.H * self.W * nz, conv_kernel, conv_stride, 
                       conv_padding, bias=False, groups=self.H * self.W),
            nn.BatchNorm2d(self.H * self.W * nz),
            nn.LeakyReLU(0.2, inplace=True))
        self.fold = nn.Fold(output_size=(hlatent, wlatent),
                            kernel_size=(unfold_kernel // conv_stride, unfold_kernel // conv_stride),
                            stride=unfold_stride // conv_stride)

    def forward(self, input):
        batch_size = input.shape[0]
        nc = self.nc
        nz = self.nz
        h = self.h
        w = self.w
        conv_stride = self.conv_stride
        unfold_kernel = self.unfold_kernel
        H = self.H
        W = self.W

        # input size N x nc x h x w
        x = self.unfold(input).reshape(batch_size, nc, unfold_kernel * unfold_kernel, H * W)
        x = x.permute(0, 3, 1, 2).reshape(batch_size, H * W * nc, unfold_kernel, unfold_kernel)
        x = self.conv(x)
        # x size is N x (H*W*nz) x k/stride x k/stride
        x = x.reshape(batch_size, H * W, nz * (unfold_kernel // conv_stride) * (unfold_kernel // conv_stride)).permute(0, 2, 1)
        # size is batch_size,nz*(k/s)*(k/s), H*W
        output = self.fold(x)
        # x size is N x nz x hlatent x wlatent
        return output


class Encoder512_F(nn.Module):
    # implementation of a 512x512 encoder using 3 layers using patchs and 4 standard conv layers
    def __init__(self, nc=nc, nef=8, nz=nz):
        nf = nef        # number of channels of the intermediate layers
        self.nz = nz    # number of latent outputs
        super().__init__()

        self.main = nn.Sequential(
            Encoder_Layer_F(nc=nc, nz=nf, h=512, w=512, hlatent=256, wlatent=256, unfold_kernel=64,
                            unfold_stride=64, conv_stride=2, conv_padding=1, conv_kernel=4),

            Encoder_Layer_F(nc=nf, nz=nf, h=256, w=256, hlatent=128, wlatent=128, unfold_kernel=64,
                            unfold_stride=64, conv_stride=2, conv_padding=1, conv_kernel=4),

            Encoder_Layer_F(nc=nf, nz=nf, h=128, w=128, hlatent=64, wlatent=64, unfold_kernel=64,
                            unfold_stride=64, conv_stride=2, conv_padding=1, conv_kernel=4),
            nn.Conv2d(nf, nf, 4, 2, 1, bias=False),
            # N x nf x 32 x 32
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf, 4, 2, 1, bias=False),
            # N x nf x 16 x 16
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf, 4, 2, 1, bias=False),
            # N x nf x 8 x 8
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf, 4, 2, 1, bias=False),
            # N x nf x 4 x 4
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nz, 4, 1, 0, bias=False))

    def forward(self, input):
        batch_size = input.shape[0]
        return self.main(input).reshape(batch_size, self.nz)


class Decoder_Layer_F(nn.Module):
    def __init__(self, nc, nz, h, w, hlatent, wlatent, unfold_kernel, unfold_stride, conv_stride, conv_padding,
                 conv_kernel, last_layer=False):
        super().__init__()
        self.nz = nz # number of channels of input latents
        self.nc = nc # number of channels of the output
        self.h = h  # input image height
        self.w = w  #

        assert hlatent == h / conv_stride
        assert wlatent == w / conv_stride

        self.hlatent = hlatent  # height of input latent image
        self.wlatent = wlatent
        self.unfold_kernel = unfold_kernel  # size of patchs
        self.unfold_stride = unfold_stride  # stride of patchs

        assert unfold_kernel % conv_stride == 0

        self.conv_stride = conv_stride    # stride of conv layer

        assert h % unfold_stride == 0
        assert w % unfold_stride == 0

        self.H = h // unfold_stride   # H number of patchs in the vertical dim
        self.W = w // unfold_stride   # H*W = total number of patchs
        self.last_layer = last_layer  # If we are on the last layer, we do not apply BatchNorm

        self.unfold = nn.Unfold(unfold_kernel//conv_stride, stride=unfold_stride//conv_stride)
        self.convtranspose = nn.ConvTranspose2d(self.H * self.W * nz, self.H * self.W * nc, conv_kernel, conv_stride, conv_padding, bias=False, groups=self.H * self.W)
        self.batchrelu = nn.Sequential(nn.BatchNorm2d(self.H * self.W * nz),
                                        nn.LeakyReLU(0.2, inplace=True))
        self.fold = nn.Fold(output_size=(h, w),
                            kernel_size=(unfold_kernel, unfold_kernel),
                            stride=unfold_stride)

    def forward(self, input):
        batch_size = input.shape[0]
        nc = self.nc
        nz = self.nz
        h = self.h
        w = self.w
        conv_stride = self.conv_stride
        unfold_kernel = self.unfold_kernel
        H = self.H
        W = self.W
        last_layer = self.last_layer

        # input size N x nc x hlatent x wlatent
        x = self.unfold(input).reshape(batch_size, nz, (unfold_kernel//conv_stride) * (unfold_kernel//conv_stride), H * W)
        x = x.permute(0, 3, 1, 2).reshape(batch_size, H * W * nz, unfold_kernel//conv_stride, unfold_kernel//conv_stride)
        x = self.convtranspose(x)
        if last_layer == False :
            x = self.batchrelu(x)
        # x size is N x (H*W*nc) x k x k
        x = x.reshape(batch_size, H * W, nc * (unfold_kernel) * (unfold_kernel)).permute(0,2,1)
        # size is batch_size,nc*k*k, H*W
        output = self.fold(x)
        # x size is N x nc x h x w
        return output


class Decoder512_F(nn.Module):
    def __init__(self,nc=nc, ngf=8, nz=nz ):
            nf=ngf
            self.nc = nc
            self.nz = nz
            super().__init__()
            self.main= nn.Sequential(
                        nn.ConvTranspose2d(nz, nf, 4, 1, 0, bias=False),
                        # N x nf x 4 x 4
                        nn.BatchNorm2d(nf),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.ConvTranspose2d(nf, nf, 4, 2, 1, bias=False),
                        # N x nf x 8 x 8
                        nn.BatchNorm2d(nf),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.ConvTranspose2d(nf, nf, 4, 2, 1, bias=False),
                        # N x nf x 16 x 16
                        nn.BatchNorm2d(nf),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.ConvTranspose2d(nf, nf, 4, 2, 1, bias=False),
                        # N x nf x 32 x 32
                        nn.BatchNorm2d(nf),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.ConvTranspose2d(nf, nf, 4, 2, 1, bias=False),
                        # N x nf x 64 x 64
                        Decoder_Layer_F(nc=nf, nz=nf, h=128, w=128, hlatent=64, wlatent=64, 
                                             unfold_kernel=64, unfold_stride=64, conv_stride=2, 
                                             conv_padding=1, conv_kernel=4),
                        Decoder_Layer_F(nc=nf, nz=nf, h=256, w=256, hlatent=128, wlatent=128, 
                                             unfold_kernel=64, unfold_stride=64, conv_stride=2, 
                                             conv_padding=1, conv_kernel=4),
                        Decoder_Layer_F(nc=nc, nz=nf, h=512, w=512, hlatent=256, wlatent=256, 
                                             unfold_kernel=64, unfold_stride=64, conv_stride=2, 
                                             conv_padding=1, conv_kernel=4, last_layer = True),
                        nn.Tanh() #nn.Sigmoid()
                        )

    def forward(self, input):
            batch_size = input.shape[0]
            return self.main(input.reshape(batch_size,self.nz,1,1)).reshape(batch_size, self.nc, 512, 512)
