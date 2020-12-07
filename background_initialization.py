
import torch
import torch.nn as nn

from Param import image_size_H, image_size_W, nc, nf, batch_size

class BENet(nn.Module):
    def __init__(self,nf=nf, nc=nc):
        super(BENet, self).__init__()
        self.nf = nf
        self.nc = nc

        self.convLayer = nn.Conv2d(1, self.nf, 3, 1, 1, bias=True)
        self.m = nn.MaxPool2d((3,3), stride=1, padding=1)

    def forward(self, input):
        # input: batch_size x nc x h x w
        input_frames = input.permute(1,0,2,3)
        # input_frames: nc x batch_size x h x w
        x = torch.mean(input_frames,dim=1)  # average pooling
        x = x.unsqueeze(1)
        # nc x 1 x h x w
        x = self.convLayer(x)
        # nc x 8 x h x w
        x = self.m(x) # Max pooling
        # nc x 8 x h x w
        x = torch.mean(x,dim=1)  # average pooling
        x = x.unsqueeze(1)
        # nc x 1 x h x w
        x = x.permute(1,0,2,3)
        # 1 x nc x h x w
        x = x.expand(batch_size, self.nc, image_size_H, image_size_W)
        # x: batch_size x nc x h x w
 
        background = x   
        foreground = torch.abs(input - background)

        return background, foreground

class Initialize(nn.Module):
    def __init__(self):
        super(Initialize, self).__init__()


    def forward(self, input):
        N = input.shape[0]

        bg = torch.sum(input, dim=0)
        bg = bg.unsqueeze(0)
        bg = bg.expand(input.shape)

        fg = torch.abs(input - bg)
        return bg, fg



class Initialize2(nn.Module):
    def __init__(self):
        super(Initialize2, self).__init__()

    def forward(self, input):
        print('-----------------------------------------------------------------')

        I_ = input.permute(3,2,1,0) # W x H x nc X N 
        I_ = I_.byte()

        out = torch.zeros(input[0].shape)
        out = out.permute(2,1,0)

        for i in range(I_.shape[0]):
            for j in range(I_.shape[1]):
                for k in range(I_.shape[2]):
                    print(i,j,k)
                    
                    out[i,j,k] = torch.bincount(I_[i,j,k]).argmax()

        out = out.float()
        out = out.unsqueeze(3)
        out = out.permute(3,2,1,0)

        bg = out.expand(input.shape)
        fg = torch.abs(input - bg)

        return bg, fg










