# model of autoencodeur with attention and instance detection
# encoder takes as inputs images in format batch_size x nchannel x h x w and produces nze = 2 position latents
# per attention head and  nzf feature latents per attention head
# Generator performs the inverse operation

#last modification june 29th 2020 21h43

import torch
import torch.nn as nn
import math

#from Param import device

class Complex_Attention_Autoencoder(nn.Module):

    def __init__(self, nze,nzf,nc, nce,nheads, h, w,
                 dim_encoding,ncg,ncl, mask_head_flag,masked_head):
        super().__init__()

        self.h = h  # image height
        self.w = w  # image width
        self.nc = nc  # number of channels of the input image
        self.nheads = nheads # number of heads and attention maps

        self.nz_encoder = nze  # dimension per head of latent position vector
        self.nz_feature = nzf # dimension per head of latent feature vector

        self.nce = nce  # nb of channels in the encoder attention CNN
        self.ncl = ncl # nb of latent channels in the generator attention branch
        self.ncg = ncg  # nb of channels in the generator intermediate CNN ( before size expansion)

        self.dim_encoding = dim_encoding      # dimension of the positional encoding table
        self.mask_head_flag = mask_head_flag  # set to True if we want to erase the 
                                              # associated object detection ( used for 
                                              # mask generation)
        self.masked_head = masked_head          # head number to erase
        nlayers = 3
        self.nlayers = nlayers    # number of layers of the U-net
        nlg = 4
        self.nlg = nlg  # number of layers of the generator final CNN


        # construction of positional encoding table
        pi = 3.1415927410125732
        assert dim_encoding % 4 == 2

        positional_encoding = torch.zeros((1, dim_encoding, h, w))
        for x in range(w):
            for y in range(h):
                xn = (x / w - 0.5)
                yn = (y / h - 0.5)
                positional_encoding[0, 0 , y, x] = xn
                positional_encoding[0, 1, y, x] = yn
                for i in range((dim_encoding-2) // 4):
                    xm = xn * (2 ** i)*pi
                    ym = yn * (2 ** i)*pi
                    positional_encoding[0, 2+0 + 4 * i, y, x] = math.sin(xm)
                    positional_encoding[0, 2+1 + 4 * i, y, x] = math.sin(ym)
                    positional_encoding[0, 2+2 + 4 * i, y, x] = math.cos(xm)
                    positional_encoding[0, 2+3 + 4 * i, y, x] = math.cos(ym)

        self.positional_encoding = nn.Parameter(positional_encoding, requires_grad=False)
        print('created fixed positional encoding table with ', h * w * dim_encoding, ' parameters')

        # encoder  layers :
        # -U-net to compute attention logits and feature latent maps
        #  -value parameters (spatial dependent, not learnable) to translate attention map to position latents
        #

        # U-net

        self.first_feature_map = nn.Sequential(nn.Conv2d(nc+dim_encoding, nce, 3, 1, 1, bias=False),
                      nn.BatchNorm2d(nce),
                      nn.LeakyReLU(0.2, inplace=True))

        nfe= [nce*(2**i) for i in range(nlayers+1)] # number of features at each layer og U-net

        self.encoder_down = nn.ModuleList([nn.Sequential(
                          nn.Conv2d(nfe[i],nfe[i+1] , 4, 2, 1, bias=False),
                          nn.BatchNorm2d(nfe[i+1]),
                          nn.LeakyReLU(0.2, inplace=True),
                          ) for i in range(nlayers)])

        self.encoder_mix = nn.Sequential(
                          nn.Conv2d(nfe[nlayers],nfe[nlayers] , 3, 1, 1, bias=False),
                          nn.BatchNorm2d(nfe[nlayers]),
                          nn.LeakyReLU(0.2, inplace=True))


        self.encoder_first_up = nn.Sequential(
                          nn.ConvTranspose2d(nfe[nlayers],nfe[nlayers-1] , 4, 2, 1, bias=False),
                          nn.BatchNorm2d(nfe[nlayers-1]),
                          nn.LeakyReLU(0.2, inplace=True))

        self.encoder_up = nn.ModuleList([nn.Sequential(
                      nn.ConvTranspose2d(2*nfe[i+1],nfe[i] , 4, 2, 1, bias=False),
                      nn.BatchNorm2d(nfe[i]),
                      nn.LeakyReLU(0.2, inplace=True)) for i in range(0,nlayers-1)])


        self.encoder_top = nn.Sequential(
                          nn.Conv2d(2*nfe[0],nfe[0] , 3, 1, 1, bias=False),
                          nn.BatchNorm2d(nfe[0]),
                          nn.LeakyReLU(0.2, inplace=True))


        self.encoder_attention_logits = nn.Conv2d(nce, nheads, 3, 1, 1, bias=False)

        self.feature_latent_map = nn.Conv2d(nce, nzf, 3, 1, 1, bias=False)

        # position value parameter table used to compute position latents

        position_latent_map = torch.zeros((nze, h, w))
        assert nze == 2
        for x in range(w):
            for y in range(h):
                position_latent_map[0, y, x] = (x / w - 0.5)
                position_latent_map[1, y, x] = (y / h - 0.5)
        self.position_latent_map = nn.Parameter(position_latent_map.reshape(1, 1, nze, h * w), requires_grad=False)

        print('encoder network created')

        # generator layers :    -CNN for attention logit computation,necessary to produce 
                                # the attention maps for each head
        #
        #                       -CNN for final image production using the object feature 
                                # latents and the attention map and the background



        # CNN for attention logit computation, takes as input the position latents of each head and the positional encoding table

        self.generator_attention_logits1 = nn.Sequential(
                            nn.Conv2d(((nheads*nze)+dim_encoding), ncl, 5, 1, 2, bias=False),
                            nn.BatchNorm2d(ncl),
                            nn.LeakyReLU(0.2, inplace=True))
        self.generator_attention_logits2 = nn.Sequential(
                            nn.Conv2d(ncl, ncl, 5, 1, 2, bias=False),
                            nn.BatchNorm2d(ncl),
                            nn.LeakyReLU(0.2, inplace=True))
        self.generator_attention_logits3 = nn.Conv2d(ncl, nheads, 5, 1, 2, bias=False)


        # CNN for image generation, takes as input a feature map with nzf channels computed from the nzf latents and the attention maps

        self.first_outputlayer = nn.Conv2d(nzf, ncg, 3, 1, 1, bias=False)


        self.outputlayers = nn.ModuleList([nn.Sequential(nn.BatchNorm2d(ncg),
                                                         nn.LeakyReLU(0.2, inplace=True),
                                                         nn.Conv2d(ncg, ncg, 7, 1, 3, bias=False)) for i in range(nlg)])


        self.last_layer = nn.Sequential(nn.BatchNorm2d(ncg),
                                           nn.LeakyReLU(0.2, inplace=True),
                                           nn.Conv2d(ncg, nc, 3, 1, 1, bias=False))
        print('Generator created')

    def encoder(self,input):

        # input shape  is batch_size x nc x h x w
        h = self.h
        w = self.w
        batch_size = input.shape[0]
        assert input.shape == (batch_size, self.nc, h, w)
        nheads = self.nheads
        dim_encoding = self.dim_encoding
        nze = self.nz_encoder
        nzf = self.nz_feature
        nlayers = self.nlayers
        x = input


        expanded_positional_encoding = self.positional_encoding.expand(batch_size, dim_encoding, h, w)
        x = torch.cat((x,expanded_positional_encoding), dim=1)
        x = self.first_feature_map(x) #

        # U-Net
        pyramid = []
        for i in range(nlayers):
            pyramid.append(x)
            x = self.encoder_down[i](x)

        x = self.encoder_mix(x)+x
        x = self.encoder_first_up(x)

        for i in reversed(range(0,nlayers-1)):
            x = self.encoder_up[i](torch.cat([x,pyramid[i+1]], dim = 1))

        x = self.encoder_top(torch.cat([x,pyramid[0]], dim = 1))

        # computation of attention maps ( one for each head)

        attention_logits = self.encoder_attention_logits(x).reshape(batch_size, nheads,h*w)
        attention_maps = nn.functional.softmax(attention_logits, dim = 2).reshape(batch_size, nheads,h,w)

        #option to disable one encoder attention head used to build object masks associated to this head
        if self.mask_head_flag == True:
            attention_maps[:,self.masked_head,:,:] = torch.zeros(batch_size,h,w)

        # computation of position latents
        expanded_attention_maps = attention_maps.reshape(batch_size,nheads,1,h*w)\
            .expand(batch_size,nheads, nze, h*w)

        expanded_position_latent_map = self.position_latent_map.expand(batch_size, nheads, nze, h*w)
        position_latents = torch.sum(expanded_attention_maps * expanded_position_latent_map, dim=3)

        # computation of feature latents
        expanded_attention_maps = attention_maps.reshape(batch_size, nheads, 1, h * w) \
            .expand(batch_size, nheads, nzf, h * w).detach()  # detach necessary for localisation convergence

        feature_latent_map  = self.feature_latent_map(x)
        expanded_feature_latent_map = feature_latent_map.reshape(batch_size, 1, nzf, h * w).expand(batch_size, nheads, nzf, h * w)

        feature_latents = torch.sum( expanded_attention_maps * expanded_feature_latent_map, dim = 3)

        objects_latents= torch.cat([position_latents, feature_latents], dim = 2)

        assert objects_latents.shape == (batch_size, nheads, nze+nzf)

        return objects_latents, attention_maps

    def generator(self, input):

        # input size is batch_size x nheadse x (nze+nzf)

        h = self.h
        w = self.w
        nze= self.nz_encoder
        nzf = self.nz_feature
        nheads = self.nheads
        dim_encoding = self.dim_encoding


        objects_latents = input
        position_latents = objects_latents[:, :, :nze]
        feature_latents = objects_latents[:, :, nze:]
        batch_size = position_latents.shape[0]

        assert position_latents.shape == (batch_size, nheads, nze)
        assert feature_latents.shape == (batch_size, nheads, nzf)

        # computation of attention maps using the position latents

        expanded_position_latents = position_latents.reshape(batch_size, nheads*nze, 1, 1).expand(batch_size, nheads*nze, h, w)
        expanded_positional_encoding = self.positional_encoding.reshape(1,dim_encoding, h, w).\
            expand(batch_size,dim_encoding, h, w)

        x = torch.cat([expanded_position_latents, expanded_positional_encoding], dim = 1).reshape(batch_size,nheads*nze+dim_encoding, h, w)

        x = self.generator_attention_logits1(x)
        x = self.generator_attention_logits2(x) + x # skip connection
        attention_logits = self.generator_attention_logits3(x).reshape(batch_size, nheads,h * w)

        attention_maps = nn.functional.softmax(attention_logits, dim=2).reshape(batch_size, nheads, h , w)

        # computation of feature map with nzf channels using attention maps and feature latents

        expanded_attention_maps = attention_maps.reshape(batch_size, nheads, 1, h,w)\
            .expand(batch_size, nheads, nzf, h, w)

        expanded_feature_latents = feature_latents.reshape(batch_size, nheads, nzf,1,1).expand(batch_size, nheads, nzf,h,w)
        x = torch.sum(expanded_attention_maps * expanded_feature_latents, dim = 1).reshape(batch_size, nzf,h,w)

        # CNN for computation of final image from feature map

        x = self.first_outputlayer(x)

        for i in range(self.nlg):
            x = self.outputlayers[i](x) + x
        images = self.last_layer(x)

        return images, attention_maps

    def forward(self, input):

        batch_size = input.shape[0]
        if input.shape == (batch_size, self.nheads, self.nz_encoder+self.nz_feature):
            output = self.generator(input)
        elif input.shape == (batch_size, self.nc, self.h, self.w) :
            output = self.encoder(input)
        else:
            print('wrong input format {input.shape}')
            exit(0)
        return output
