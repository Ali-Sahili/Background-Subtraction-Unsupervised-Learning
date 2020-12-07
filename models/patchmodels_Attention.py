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
        self.batchrelu = nn.Sequential(nn.BatchNorm2d(self.H * self.W * nc),
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




class Encoder_Decoder512(nn.Module):
    # implementation of a 512x512 auto-encoder using 3 layers using patchs and 4 standard conv layers for encoder and the same for decoder
    def __init__(self, nc=nc, nef=8, ngf=8, nz=nz):
        nf = nef        # number of channels of the intermediate layers in encoder
        nfd = ngf       # number of channels of the intermediate layers in decoder
        self.nz = nz    # number of latent outputs
        self.nc = nc    # number of channels
        super().__init__()

        #---------------------
        #    Encoder Part 
        #---------------------
        self.Encoder_layer1 = Encoder_Layer_F( nc=nc, nz=nf*2, h=512, w=512, hlatent=256, wlatent=256, 
                                     unfold_kernel=64, unfold_stride=64, conv_stride=2, 
                                     conv_padding=1, conv_kernel=4)

        self.Encoder_layer2 = Encoder_Layer_F( nc=nf*2, nz=nf*4, h=256, w=256, hlatent=128, wlatent=128, 
                                       unfold_kernel=64, unfold_stride=64, conv_stride=2, 
                                       conv_padding=1, conv_kernel=4)

        self.Encoder_layer3 = Encoder_Layer_F( nc=nf*4, nz=nf*8, h=128, w=128, hlatent=64, wlatent=64, 
                                       unfold_kernel=64, unfold_stride=64, conv_stride=2, 
                                       conv_padding=1, conv_kernel=4)


        self.Encoder_layer4 = nn.Conv2d(nf*8, nf*16, 4, 2, 1, bias=False)

        self.Encoder_layer5 = nn.Sequential(
                                # N x nf x 32 x 32
                                nn.BatchNorm2d(nf*16),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(nf*16, nf*16, 4, 2, 1, bias=False))

        self.Encoder_layer6 = nn.Sequential(
                                # N x nf x 16 x 16
                                nn.BatchNorm2d(nf*16),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(nf*16, nf*16, 4, 2, 1, bias=False))

        self.Encoder_layer7 = nn.Sequential(
                                # N x nf x 8 x 8
                                nn.BatchNorm2d(nf*16),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(nf*16, nf*16, 4, 2, 1, bias=False))

        self.Encoder_layer8 = nn.Sequential(
                                # N x nf x 4 x 4
                                nn.BatchNorm2d(nf*16),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(nf*16, nz, 4, 1, 0, bias=False))


        #---------------------
        #    Decoder Part 
        #---------------------
        self.Decoder_layer1 = nn.ConvTranspose2d(nz, nfd*16, 4, 1, 0, bias=False)

        self.Decoder_layer2 = nn.Sequential(
                        # N x nf x 4 x 4
                        nn.BatchNorm2d(nfd*16),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.ConvTranspose2d(nfd*16, nfd*16, 4, 2, 1, bias=False))

        self.Decoder_layer3 = nn.Sequential(
                        # N x nf x 8 x 8
                        nn.BatchNorm2d(nfd*16),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.ConvTranspose2d(nfd*16, nfd*16, 4, 2, 1, bias=False))

        self.Decoder_layer4 = nn.Sequential(
                        # N x nf x 16 x 16
                        nn.BatchNorm2d(nfd*16),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.ConvTranspose2d(nfd*16, nfd*16, 4, 2, 1, bias=False))

        self.Decoder_layer5 = nn.Sequential(
                        # N x nf x 32 x 32
                        nn.BatchNorm2d(nfd*16),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.ConvTranspose2d(nfd*16, nfd*8, 4, 2, 1, bias=False))

                        # N x nf x 64 x 64
        self.Decoder_layer6 = Decoder_Layer_F(nc=nfd*4, nz=nfd*8, h=128, w=128, hlatent=64, wlatent=64, 
                                             unfold_kernel=64, unfold_stride=64, conv_stride=2, 
                                             conv_padding=1, conv_kernel=4)

        self.Decoder_layer7 = Decoder_Layer_F(nc=nfd*2, nz=nfd*4, h=256, w=256, hlatent=128, wlatent=128, 
                                             unfold_kernel=64, unfold_stride=64, conv_stride=2, 
                                             conv_padding=1, conv_kernel=4)

        self.Decoder_layer8 = Decoder_Layer_F(nc=nc, nz=nfd*2, h=512, w=512, hlatent=256, wlatent=256, 
                                             unfold_kernel=64, unfold_stride=64, conv_stride=2, 
                                             conv_padding=1, conv_kernel=4, last_layer = True)
        self.Decoder_layer9 = nn.Tanh() #nn.Sigmoid()


        #------------------------
        #    Attention Modules 
        #------------------------

        self.attention_layer = Attention_layer(nf)
        self.attention_layer2 = Attention_layer(nf*2)
        self.attention_layer4 = Attention_layer(nf*4)
        self.attention_layer8 = Attention_layer(nf*8)
        self.attention_layer16 = Attention_layer(nf*16)

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

        # N x nf*2 x 256 x 256
        x2 = self.Encoder_layer2(x1)
        m_x2 = self.Encoder_layer2(m_x1)

        # N x nf*4 x 128 x 128
        x3 = self.Encoder_layer3(x2)
        m_x3 = self.Encoder_layer3(m_x2)

        # N x nf*8 x 64 x 64
        x4 = self.Encoder_layer4(x3)
        m_x4 = self.Encoder_layer4(m_x3)

        # N x nf*16 x 32 x 32
        x5 = self.Encoder_layer5(x4)
        m_x5 = self.Encoder_layer5(m_x4)

        # N x nf*16 x 16 x 16
        x6 = self.Encoder_layer6(x5)
        m_x6 = self.Encoder_layer6(m_x5)

        # N x nf*16 x 8 x 8
        x7 = self.Encoder_layer7(x6)
        m_x7 = self.Encoder_layer7(m_x6)

        # N x nf*16 x 4 x 4
        Encoder_out = self.Encoder_layer8(x7)
        m_Encoder_out = self.Encoder_layer8(m_x7)

        # N x nz x 1 x 1
        Encoder_out = Encoder_out.reshape(batch_size, self.nz, 1, 1)
        m_Encoder_out = m_Encoder_out.reshape(batch_size, self.nz, 1, 1)



        #########################
        #      Decoding
        #########################
        # N x nz x 1 x 1
        y1 = self.Decoder_layer1(Encoder_out)
        m_y1 = self.Decoder_layer1(m_Encoder_out)
        weight1 = self.attention_layer16(m_x7, m_y1)

        # N x nf*16 x 4 x 4
        a1 = torch.mul(x7, weight1) + y1
        y2 = self.Decoder_layer2(a1)
        m_y2 = self.Decoder_layer2(m_y1)
        weight2 = self.attention_layer16(m_x6, m_y2)

        # N x nf*16 x 8 x 8
        a2 = torch.mul(x6, weight2) + y2
        y3 = self.Decoder_layer3(a2)
        m_y3 = self.Decoder_layer3(m_y2)
        weight3 = self.attention_layer16(m_x5, m_y3)

        # N x nf*16 x 16 x 16
        a3 = torch.mul(x5, weight3) + y3
        y4 = self.Decoder_layer4(a3)
        m_y4 = self.Decoder_layer4(m_y3)
        weight4 = self.attention_layer16(m_x4,m_y4)

        # N x nf*16 x 32 x 32
        a4 = torch.mul(x4, weight4) + y4
        y5 = self.Decoder_layer5(a4)
        m_y5 = self.Decoder_layer5(m_y4)
        weight5 = self.attention_layer8(m_x3,m_y5)

        # N x nf*8 x 64 x 64
        a5 = torch.mul(x3, weight5) + y5
        y6 = self.Decoder_layer6(a5)
        m_y6 = self.Decoder_layer6(m_y5)
        weight6 = self.attention_layer4(m_x2,m_y6)

        # N x nf*4 x 128 x 128
        a6 = torch.mul(x2, weight6) + y6
        y7 = self.Decoder_layer7(a6)
        m_y7 = self.Decoder_layer7(m_y6)
        weight7 = self.attention_layer2(m_x1,m_y7)

        # N x nf*2 x 256 x 256
        a7 = torch.mul(x1, weight7) + y7
        y8 = self.Decoder_layer8(a7)
        m_y8 = self.Decoder_layer8(m_y7)

        # N x nc x 512 x 512
        out = self.Decoder_layer9(y8)
        m_out = self.Decoder_layer9(m_y8)





        return out.reshape(batch_size, self.nc, 512, 512)

    



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
        x = torch.mul(Dec, y)  # pixel-wise multiplication
        
        #out = torch.cat((x,Dec), dim=1) # dimension: n*2 x h x w
        out = x + Dec
        return out.view(batch_size, n, h, w)
        









