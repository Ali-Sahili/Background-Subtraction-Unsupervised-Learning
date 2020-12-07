import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

from modules import conv_block, MultiScaleHGResidual, SoftArgmax2D, Hourglass, SlotAttention
from modules import get_positional_encoding

from Param import device

def latent_to_mean_std(latent_var):
    '''
    Converts a VAE latent vector to mean and std. log_std is converted to std.
    :param latent_var: VAE latent vector
    :return:
    '''
    mean, log_std = torch.chunk(latent_var, 2, dim=-1)
    # std = log_std.mul(0.5).exp_()
    std = torch.sigmoid(log_std.clamp(-10, 10)) * 2
    return mean, std


def stn(image, z_where, output_dims, device, inverse=False):
    """
    Slightly modified based on https://github.com/kamenbliznashki/generative_models/blob/master/air.py

    spatial transformer network used to scale and shift input according to z_where in:
            1/ x -> x_att   -- shapes (H, W) -> (attn_window, attn_window) -- thus inverse = False
            2/ y_att -> y   -- (attn_window, attn_window) -> (H, W) -- thus inverse = True
    inverting the affine transform as follows: A_inv ( A * image ) = image
    A = [R | T] where R is rotation component of angle alpha, T is [tx, ty] translation component
    A_inv rotates by -alpha and translates by [-tx, -ty]
    if x' = R * x + T  -->  x = R_inv * (x' - T) = R_inv * x - R_inv * T
    here, z_where is 3-dim [scale, tx, ty] so inverse transform is [1/scale, -tx/scale, -ty/scale]
    R = [[s, 0],  ->  R_inv = [[1/s, 0],
         [0, s]]               [0, 1/s]]
    """

    xt, yt, xs, ys = torch.chunk(z_where, 4, dim=-1)
    yt = yt.squeeze()
    xt = xt.squeeze()
    ys = ys.squeeze()
    xs = xs.squeeze()

    batch_size = image.shape[0]
    color_chans = image.shape[1]
    out_dims = [batch_size, color_chans] + output_dims # [Batch, RGB, obj_h, obj_w]

    # Important: in order for scaling to work, we need to convert from top left corner of bbox to center of bbox
    yt = (yt ) * 2 - 1
    xt = (xt ) * 2 - 1

    theta = torch.zeros(2, 3).repeat(batch_size, 1, 1).to(device)
    #print(z_where.shape, xs.shape)

    # set scaling
    theta[:, 0, 0] = xs
    theta[:, 1, 1] = ys
    # set translation
    theta[:, 0, -1] = xt
    theta[:, 1, -1] = yt

    # inverse == upsampling
    if inverse:
        # convert theta to a square matrix to find inverse
        t = torch.tensor([0., 0., 1.]).repeat(batch_size, 1, 1).to(device)
        t = torch.cat([theta, t], dim=-2)
        t = t.inverse()
        theta = t[:, :2, :]
        out_dims = [batch_size, color_chans + 1] + output_dims  # [Batch, RGBA, obj_h, obj_w]

    # 2. construct sampling grid
    grid = F.affine_grid(theta, out_dims).to(device)

    # 3. sample image from grid
    padding_mode = 'border' if not inverse else 'zeros'
    input_glimpses = F.grid_sample(image, grid, padding_mode=padding_mode)
    # debug_tools.plot_stn_input_and_out(input_glimpses)


    return input_glimpses


class HourglassNet(nn.Module):
    def __init__(self, n_inputs=3, n_outputs=6, bw=64, hg_depth=4, upmode='bilinear', 
                           use_drop=False, in_verbose = False, out_verbose=False):

        super(HourglassNet, self).__init__()

        self.out_verbose = out_verbose

        self.layer1 = nn.Sequential(
            conv_block(n_inputs, bw, ks=7, stride=2, padding=3, activation='relu', bias=False),
            MultiScaleHGResidual(bw, bw * 2),
            conv_block(bw*2, bw*2, ks=4, stride=2, padding=1, activation='relu', bias=False)
        )

        self.layer2 = nn.Sequential( MultiScaleHGResidual(bw * 2, bw * 2),
                                     MultiScaleHGResidual(bw * 2, bw * 2),
                                     MultiScaleHGResidual(bw * 2, bw * 4)
                                   )

        self.layers = []
        self.hourglass = Hourglass( hg_depth, bw * 4, bw * 4, bw * 8, upmode,
                                    verbose=in_verbose, layers=self.layers)

        self.mixer = nn.Sequential(nn.Dropout2d(p=0.25),
                                   conv_block(bw * 8, bw * 8, 1, 1, 0),
                                   nn.Dropout2d(p=0.25),
                                   conv_block(bw * 8, bw * 4, 1, 1, 0))

        self.out_block = nn.Sequential( nn.ConvTranspose2d(bw *4, bw*2, 4, 2, 1),
                                        nn.BatchNorm2d(bw*2),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(bw*2, bw, 4, 2, 1),
                                        nn.BatchNorm2d(bw),
                                        nn.ReLU()
                                       )

    def forward(self, x):

        o_layer_1 = self.layer1(x)
        o_layer_2 = self.layer2(o_layer_1)
        o_hg, inter_layers = self.hourglass(o_layer_2)
        o_mixer = self.mixer(o_hg)
        out = self.out_block(o_mixer)

        if self.out_verbose: 
            print('Input shape:      ', x.shape)
            print('Before hourglass: ', o_layer_1.shape)
            print('Before hourglass: ', o_layer_2.shape)
            print('output hourglass: ', o_hg.shape)
            print('out block:        ', o_mixer.shape)
            print('Output:           ', out.shape)
            print()
            print('Lower part of each hourglass network')
            for i in range(len(inter_layers)):
                print(inter_layers[i].shape)
            print()
            print()

        return out, o_mixer, inter_layers


class Final_Model(nn.Module):
    def __init__(self, height, width, hidden= 64, n_heads = 6, nze=2, nc=3, 
                          n_stack=4, n_iters=3, n_slots=5):
        super().__init__()

        assert height == width
        self.dim = height #image size

        self.nf = hidden 
        self.n_heads = n_heads
        self.nc = nc # number of channels
        self.n_stack = n_stack  # number of stacked hourglass networks
        self.n_iters = n_iters # number of iterations for slot attention
        self.n_slots = n_slots

        # construction of positional encoding table
        self.dim_encoding = 6
        self.positional_encoding = get_positional_encoding(self.dim_encoding, height, width, requires_grad=False, device='cpu')
        
        # Hourglass Network
        self.hourglass = HourglassNet( n_inputs=nc+self.dim_encoding, n_outputs=self.n_heads, 
                                        bw=self.nf, hg_depth=n_stack,upmode='bilinear', 
                                        use_drop=False, in_verbose = False, out_verbose=False )

        # Soft Argmax 
        self.block_sagm = nn.Sequential(nn.Conv2d(self.nf * 4, n_heads, kernel_size=1, padding=0))
        self.sagm = SoftArgmax2D()

        # Slot Attention module
        self.slot_attn = SlotAttention(num_slots = n_slots, dim = self.nf*self.nf//4, iters = n_iters)

        # position value parameter table used to compute position latents
        position_latent_map = torch.zeros((nze, height, width))
        assert nze == 2
        for x in range(width):
            for y in range(height):
                position_latent_map[0, y, x] = (x / width - 0.5)
                position_latent_map[1, y, x] = (y / height - 0.5)
        self.position_latent_map = nn.Parameter(position_latent_map.reshape(1, 1, nze, height * width), requires_grad=False)



        # Generator
        self.generator = nn.Sequential( nn.ConvTranspose2d(self.nf*4, self.nf*2, 4, 2, 1),
                                        nn.BatchNorm2d(self.nf*2),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(self.nf*2, self.nf, 4, 2, 1),
                                        nn.BatchNorm2d(self.nf),
                                        nn.ReLU(),
                                        nn.Conv2d(self.nf, self.nc, 3, 1, 1),
                                        nn.Sigmoid()
                                      )


        self.layer_what = nn.Sequential(
                 conv_block(self.nf*4, 512, ks=4, stride=2, padding=1, activation='relu', bias=False),
                 conv_block(512, 512, ks=4, stride=2, padding=1, activation='relu', bias=False),
                 conv_block(512, self.nf*4, ks=4, stride=2, padding=1, activation=None, bias=False)
                      )

        self.out_what = nn.Sequential( nn.ConvTranspose2d(72, 64, 4, 2, 1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 4, 3, 1, 1),
                                      )

    def forward(self, input, verbose=False):
        # batch size
        b_size = input.shape[0]

        # create positional encoding table
        expanded_positional_encoding = self.positional_encoding.expand(b_size, self.dim_encoding, self.dim, self.dim)
        x = torch.cat((input,expanded_positional_encoding), dim=1)

        # apply hourglass network on input image concatenated with positional encoding table
        out, features, inter_layers = self.hourglass(x)


        # z where
        throught_Features = inter_layers[0].view(b_size, -1, 8)
        mean, var = latent_to_mean_std(throught_Features)
        dist = torch.distributions.Normal(loc=mean, scale=var)
        z_where = dist.rsample()
        if verbose: print('Z where ', z_where.shape)

        tmp = []
        anchor_boxes = [32,32]
        for k in range(z_where.shape[1]):
            a = stn(x, z_where[:,k], output_dims=anchor_boxes, device=device, inverse=False)
            tmp.append(a)
        stn_encode_out = torch.cat(tmp, dim=1).view(b_size, z_where.shape[1],self.dim_encoding+self.nc,anchor_boxes[0],anchor_boxes[1]).view(b_size, z_where.shape[1], anchor_boxes[0]*3, anchor_boxes[1]*3)
        if verbose: print('stn_encode_out', stn_encode_out.shape)

        throught_Features2 = self.layer_what(stn_encode_out).view(b_size, z_where.shape[1], -1)
        mean, var = latent_to_mean_std(throught_Features2)
        dist = torch.distributions.Normal(loc=mean, scale=var)
        z_what = dist.rsample()
        if verbose: print('Z what ', z_what.shape)

        z_what_flat = z_what.view(b_size,-1, int(math.sqrt(self.nf*4)),int(math.sqrt(self.nf*4)))
        if verbose: print('Z what ', z_what_flat.shape)

        objects = self.out_what(z_what_flat)
        
        transformed_imgs = []
        for k in range(z_where.shape[1]):
            a = stn(objects, z_where[:,k], [self.dim,self.dim],  device=device, inverse=True)
            transformed_imgs.append(a)
        transformed_imgs = torch.cat(transformed_imgs, dim=1).view(b_size, -1,4,self.dim,self.dim)
        imgs = torch.mean(transformed_imgs, dim=1).view(b_size, -1,self.dim,self.dim)
        if verbose: print(imgs.shape)

        weights = torch.cat([imgs[:,3:,:,:], imgs[:,3:,:,:], imgs[:,3:,:,:]], dim=1)
        out_img = imgs[:,:3,:,:]
        out_img = torch.matmul(out_img, weights)
        if verbose: print('Finishing', out_img.shape, weights.shape)


        # Soft Argmax
        #b_sagm = self.block_sagm(features)
        #o_sagm = self.sagm(b_sagm)
        #print('Before SoftArgmax:', b_sagm.shape)
        #print('After SoftArgmax: ', o_sagm.shape)

        # slot Attention mechanism
        #out_att = self.slot_attn(b_sagm.view(b_size,self.n_heads,-1)).view(b_size, self.n_slots, 
        #                                                                     self.nf//2, self.nf//2)
        #print('Attention map:', out_att.shape)




        return out_img



