import torch
import torch.nn as nn

from Param import nc, nz




def batch_ReLU_layer(nf):
    return nn.BatchNorm2d(nf), nn.LeakyReLU(0.2, inplace=True)

class Encoder256_Atrous(nn.Module):
    def __init__(self,nz=nz,nef=16,nc=nc):
        super(Encoder256_Atrous, self).__init__()
        self.nz=nz
        self.nc=nc

        self.BN_ReLU_layer_1 = nn.Sequential(*batch_ReLU_layer(nef))
        self.BN_ReLU_layer_2 = nn.Sequential(*batch_ReLU_layer(nef*2))
        self.BN_ReLU_layer_3 = nn.Sequential(*batch_ReLU_layer(nef*4))
        self.BN_ReLU_layer_4 = nn.Sequential(*batch_ReLU_layer(nef*8))
        self.BN_ReLU_layer_5 = nn.Sequential(*batch_ReLU_layer(nef*16))
        self.BN_ReLU_layer_6 = nn.Sequential(*batch_ReLU_layer(nef*32))
        self.BN_ReLU_layer_7 = nn.Sequential(*batch_ReLU_layer(nef*64))

            # input is (nc) x 258 x 256
        self.conv_layer_1 = nn.Conv2d(nc, nef, 4, 2, 1, bias=False)
        self.aconv_layer_1 = nn.Conv2d(nc, nef, 2, 2, 1, dilation=3, bias=False)# atrous conv

            # state size. (nef) x 128 x 128
        self.conv_layer_2 = nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False)
        self.aconv_layer_2 = nn.Conv2d(nef, nef * 2, 2, 2, 1, dilation=3, bias=False)# atrous conv

            # state size. (nef) x 64 x 64
        self.conv_layer_3 = nn.Conv2d(nef * 2, nef * 4, 4, 2, 1, bias=False)
        self.aconv_layer_3 = nn.Conv2d(nef * 2, nef * 4, 2, 2, 1, dilation=3, bias=False)# atrous conv

            # state size. (nef) x 32 x 32
        self.conv_layer_4 = nn.Conv2d(nef*4, nef * 8, 4, 2, 1, bias=False)
        self.aconv_layer_4 = nn.Conv2d(nef*4, nef * 8, 2, 2, 1, dilation=3, bias=False)# atrous conv

            # state size. (nef*2) x 16 x 16
        self.conv_layer_5 = nn.Conv2d(nef * 8, nef * 16, 4, 2, 1, bias=False)
        self.aconv_layer_5 = nn.Conv2d(nef*8, nef * 16, 2, 2, 1, dilation=3, bias=False)# atrous conv

            # state size. (nef*4) x 8 x 8
        self.conv_layer_6 = nn.Conv2d(nef * 16, nef * 32, 4, 2, 1, bias=False)
        self.aconv_layer_6 = nn.Conv2d(nef*16, nef * 32, 2, 2, 1, dilation=3, bias=False)# atrous conv

            # state size. (nef*8) x 4 x 4
        self.conv_layer_7 = nn.Conv2d(nef * 32, nef * 64, 4, 1, 0, bias=False)
        self.conv_layer_8 = nn.Conv2d(nef * 64, nz, 1, 1, 0, bias=True)

        self.output_layer = nn.Sigmoid()


    def forward(self, input, verbose=False):
        if verbose: print('input layer: ', input.shape)

        # (nc) x 256 x 256
        x1 = self.conv_layer_1(input)
        x2 = self.aconv_layer_1(input)
        if verbose: print('layer1: ', x1.shape, x2.shape)
        x = x1 + x2
        x = self.BN_ReLU_layer_1(x)

        # (nef) x 128 x 128
        x1 = self.conv_layer_2(x)
        x2 = self.aconv_layer_2(x)
        if verbose: print('layer2: ', x1.shape, x2.shape)
        x = x1 + x2
        x = self.BN_ReLU_layer_2(x)

        # (ne*2f) x 64 x 64
        x1 = self.conv_layer_3(x)
        x2 = self.aconv_layer_3(x)
        if verbose: print('layer3: ', x1.shape, x2.shape)
        x = x1 + x2
        x = self.BN_ReLU_layer_3(x)

        # (nef*4) x 32 x 32
        x1 = self.conv_layer_4(x)
        x2 = self.aconv_layer_4(x)
        if verbose: print('layer4: ', x1.shape, x2.shape)
        x = x1 + x2
        x = self.BN_ReLU_layer_4(x)

        # (nef*8) x 16 x 16
        x1 = self.conv_layer_5(x)
        x2 = self.aconv_layer_5(x)
        if verbose: print('layer5: ', x1.shape, x2.shape)
        x = x1 + x2
        x = self.BN_ReLU_layer_5(x)

        # (nef*16) x 8 x 8
        x1 = self.conv_layer_6(x)
        x2 = self.aconv_layer_6(x)
        if verbose: print('layer6: ', x1.shape, x2.shape)
        x = x1 + x2
        x = self.BN_ReLU_layer_6(x)

        # (nef*32) x 4 x 4
        x = self.conv_layer_7(x)
        if verbose: print('layer7: ', x.shape)
        x = self.BN_ReLU_layer_7(x)

        x = self.conv_layer_8(x)
        if verbose: print('layer8: ', x.shape)

        out = self.output_layer(x)
        if verbose: print('output layer: ', out.shape)

        return out.reshape(-1, self.nz)



class Encoder256_Atrous_fuse(nn.Module):
    def __init__(self,nz=nz,nef=16,nc=nc):
        super(Encoder256_Atrous_fuse, self).__init__()
        self.nz=nz
        self.nc=nc

        self.BN_ReLU_layer_1 = nn.Sequential(*batch_ReLU_layer(nef*2))
        self.BN_ReLU_layer_2 = nn.Sequential(*batch_ReLU_layer(nef*4))
        self.BN_ReLU_layer_3 = nn.Sequential(*batch_ReLU_layer(nef*8))
        self.BN_ReLU_layer_4 = nn.Sequential(*batch_ReLU_layer(nef*16))
        self.BN_ReLU_layer_5 = nn.Sequential(*batch_ReLU_layer(nef*32))
        self.BN_ReLU_layer_6 = nn.Sequential(*batch_ReLU_layer(nef*64))
        self.BN_ReLU_layer_7 = nn.Sequential(*batch_ReLU_layer(nef*64))

            # input is (nc) x 258 x 256
        self.conv_layer_1 = nn.Conv2d(nc, nef, 4, 2, 1, bias=False)
        self.aconv_layer_1 = nn.Conv2d(nc, nef, 2, 2, 1, dilation=3, bias=False)# atrous conv

            # state size. (nef) x 128 x 128
        self.conv_layer_2 = nn.Conv2d(nef*2, nef * 2, 4, 2, 1, bias=False)
        self.aconv_layer_2 = nn.Conv2d(nef*2, nef * 2, 2, 2, 1, dilation=3, bias=False)# atrous conv

            # state size. (nef) x 64 x 64
        self.conv_layer_3 = nn.Conv2d(nef * 4, nef * 4, 4, 2, 1, bias=False)
        self.aconv_layer_3 = nn.Conv2d(nef * 4, nef * 4, 2, 2, 1, dilation=3, bias=False)# atrous conv

            # state size. (nef) x 32 x 32
        self.conv_layer_4 = nn.Conv2d(nef*8, nef * 8, 4, 2, 1, bias=False)
        self.aconv_layer_4 = nn.Conv2d(nef*8, nef * 8, 2, 2, 1, dilation=3, bias=False)# atrous conv

            # state size. (nef*2) x 16 x 16
        self.conv_layer_5 = nn.Conv2d(nef * 16, nef * 16, 4, 2, 1, bias=False)
        self.aconv_layer_5 = nn.Conv2d(nef*16, nef * 16, 2, 2, 1, dilation=3, bias=False)# atrous conv

            # state size. (nef*4) x 8 x 8
        self.conv_layer_6 = nn.Conv2d(nef * 32, nef * 32, 4, 2, 1, bias=False)
        self.aconv_layer_6 = nn.Conv2d(nef*32, nef * 32, 2, 2, 1, dilation=3, bias=False)# atrous conv

            # state size. (nef*8) x 4 x 4
        self.conv_layer_7 = nn.Conv2d(nef * 64, nef * 64, 4, 1, 0, bias=False)
        self.conv_layer_8 = nn.Conv2d(nef * 64, nz, 1, 1, 0, bias=True)

        self.output_layer = nn.Sigmoid()


    def forward(self, input, verbose=False):
        if verbose: print('input layer: ', input.shape)

        # (nc) x 256 x 256
        x1 = self.conv_layer_1(input)
        x2 = self.aconv_layer_1(input)
        if verbose: print('layer1: ', x1.shape, x2.shape)
        x = torch.cat((x1,x2), 1)
        x = self.BN_ReLU_layer_1(x)

        # (nef) x 128 x 128
        x1 = self.conv_layer_2(x)
        x2 = self.aconv_layer_2(x)
        if verbose: print('layer2: ', x1.shape, x2.shape)
        x = torch.cat((x1,x2), 1)
        x = self.BN_ReLU_layer_2(x)

        # (ne*2f) x 64 x 64
        x1 = self.conv_layer_3(x)
        x2 = self.aconv_layer_3(x)
        if verbose: print('layer3: ', x1.shape, x2.shape)
        x = torch.cat((x1,x2), 1)
        x = self.BN_ReLU_layer_3(x)

        # (nef*4) x 32 x 32
        x1 = self.conv_layer_4(x)
        x2 = self.aconv_layer_4(x)
        if verbose: print('layer4: ', x1.shape, x2.shape)
        x = torch.cat((x1,x2), 1)
        x = self.BN_ReLU_layer_4(x)

        # (nef*8) x 16 x 16
        x1 = self.conv_layer_5(x)
        x2 = self.aconv_layer_5(x)
        if verbose: print('layer5: ', x1.shape, x2.shape)
        x = torch.cat((x1,x2), 1)
        x = self.BN_ReLU_layer_5(x)

        # (nef*16) x 8 x 8
        x1 = self.conv_layer_6(x)
        x2 = self.aconv_layer_6(x)
        if verbose: print('layer6: ', x1.shape, x2.shape)
        x = torch.cat((x1,x2), 1)
        x = self.BN_ReLU_layer_6(x)

        # (nef*32) x 4 x 4
        x = self.conv_layer_7(x)
        if verbose: print('layer7: ', x.shape)
        x = self.BN_ReLU_layer_7(x)

        x = self.conv_layer_8(x)
        if verbose: print('layer8: ', x.shape)

        out = self.output_layer(x)
        if verbose: print('output layer: ', out.shape)

        return out.reshape(-1, self.nz)


