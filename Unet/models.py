import torch
import torch.nn as nn

from Param import nz, nc


class Encoder_U(nn.Module):

    def __init__(self, nef=8, nc=nc):
        super(Encoder_U, self).__init__()
        self.nc = nc

        # 3*512*512
        self.down1 = nn.Sequential(
            nn.Conv2d(self.nc, nef, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef, nef, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef),
            nn.ReLU(inplace=True))
        self.down1_pool = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        # 16*256*256
        self.down2 = nn.Sequential(
            nn.Conv2d(nef, nef*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef*2, nef*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*2),
            nn.ReLU(inplace=True),)
        self.down2_pool = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        # 32*128*128
        self.down3 = nn.Sequential(
            nn.Conv2d(nef*2, nef*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef*4, nef*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*4),
            nn.ReLU(inplace=True),)
        self.down3_pool = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        # 64*64*64
        self.down4 = nn.Sequential(
            nn.Conv2d(nef*4, nef*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef*8, nef*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*8),
            nn.ReLU(inplace=True))
        self.down4_pool = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        # 128*32*32
        self.down5 = nn.Sequential(
            nn.Conv2d(nef*8, nef*16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*16),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef*16, nef*16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*16),
            nn.ReLU(inplace=True))
        self.down5_pool = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        # 256*16*16
        self.down6 = nn.Sequential(
            nn.Conv2d(nef*16, nef*32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*32),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef*32, nef*32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*32),
            nn.ReLU(inplace=True))
        self.down6_pool = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        # 512*8*8
        self.center = nn.Sequential(
            nn.Conv2d(nef*32, nef*64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*64),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef*64, nef*64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*64),
            nn.ReLU(inplace=True),)

            # state size. (nef*32) x 8 x 8
        self.out = nn.Sequential(
            nn.Conv2d(nef * 32, nef * 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*64) x 4 x 4
            nn.Conv2d(nef * 64, nef * 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nef * 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nef * 128, 1, 1, 1, 0, bias=True),
            nn.Sigmoid())

    def forward(self, img):
        # 3*512*512
        down1 = self.down1(img)
        down1_pool = self.down1_pool(down1)

        # 16*256*256
        down2 = self.down2(down1_pool)
        down2_pool = self.down2_pool(down2)

        # 32*128*128
        down3 = self.down3(down2_pool)
        down3_pool = self.down3_pool(down3)

        # 64*64*64
        down4 = self.down4(down3_pool)
        down4_pool = self.down4_pool(down4)

        # 128*32*32
        down5 = self.down5(down4_pool)
        down5_pool = self.down5_pool(down5)

        # 256*16*16
        down6 = self.down6(down5_pool)
        down6_pool = self.down6_pool(down6)

        # 512*8*8
        center = self.center(down6_pool)
        # 1024*8*8

        down = [down1, down2, down3, down4, down5, down6, center]

        out = self.out(down6_pool)
        # 1 x 1 x 1

        return out, down



class Decoder_U(nn.Module):

    def __init__(self, nef=8, nc=nc):
        super(Decoder_U, self).__init__()
        self.nc = nc



        # 1024*8*8
        self.upsample6 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'))
        self.up6 = nn.Sequential(
            nn.Conv2d(nef*64+nef*32, nef*32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*32),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef*32, nef*32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*32),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef*32, nef*32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*32),
            nn.ReLU(inplace=True),)

        # 512*16*16
        self.upsample5 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'))
        self.up5 = nn.Sequential(
            nn.Conv2d(nef*32+nef*16, nef*16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*16),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef*16, nef*16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*16),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef*16, nef*16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*16),
            nn.ReLU(inplace=True),)

        # 256*32*32
        self.upsample4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'))
        self.up4 = nn.Sequential(
            nn.Conv2d(nef*16+nef*8, nef*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef*8, nef*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef*8, nef*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*8),
            nn.ReLU(inplace=True),)

        # 128*64*64
        self.upsample3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'))
        self.up3 = nn.Sequential(
            nn.Conv2d(nef*8+nef*4, nef*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef*4, nef*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef*4, nef*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*4),
            nn.ReLU(inplace=True),
            )

        # 64*128*128
        self.upsample2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'))
        self.up2 = nn.Sequential(
            nn.Conv2d(nef*4+nef*2, nef*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef*2, nef*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef*2, nef*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef*2),
            nn.ReLU(inplace=True),
            )

        # 32*256*256
        self.upsample1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'))
        self.up1 = nn.Sequential(
            nn.Conv2d(nef*2+nef, nef, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef, nef, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef, nef, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nef),
            nn.ReLU(inplace=True),
            )
        # 16*512*512
        self.classifier = nn.Sequential(
                nn.Conv2d(nef, self.nc, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid(),
            )
    # 3*512*512




    def forward(self, down):


        up6 = self.upsample6(down[6])
        # 1024*16*16

        up6 = torch.cat((down[5],up6), 1)
        up6 = self.up6(up6)

        # 512*16*16
        up5 = self.upsample5(up6)
        up5 = torch.cat((down[4],up5), 1)
        up5 = self.up5(up5)

        # 256*32*32
        up4 = self.upsample4(up5)
        up4 = torch.cat((down[3],up4), 1)
        up4 = self.up4(up4)

        # 128*64*64
        up3 = self.upsample3(up4)
        up3 = torch.cat((down[2],up3), 1)
        up3 = self.up3(up3)

        # 64*128*128
        up2 = self.upsample2(up3)
        up2 = torch.cat((down[1],up2), 1)
        up2 = self.up2(up2)

        # 32*256*256
        up1 = self.upsample1(up2)
        up1 = torch.cat((down[0],up1), 1)
        up1 = self.up1(up1)

        # 16*512*512
        output = self.classifier(up1)
        # 3*512*512

        return output

