import torch
from torch import nn
import torch.nn.functional as F

import math

def conv_block(inp,out, ks=3, stride=1, padding=0, activation='relu', bias=True):
    """ ConvNet building block with different activations and normalizations support."""
    assert activation=='relu' or activation==None

    return nn.Sequential(
                      nn.Conv2d(inp, out, kernel_size=ks, stride=stride, padding=padding, bias=bias),
                      nn.BatchNorm2d(out),
                      nn.ReLU(inplace=True) if activation == 'relu' else nn.Sequential()
                        )

class MultiScaleHGResidual(nn.Module):

    def __init__(self, n_inp, n_out):
        super().__init__()

        self.scale1 = conv_block(n_inp, n_out//2, 3, 1, 1, activation='relu')
        self.scale2 = conv_block(n_out//2, n_out//4, 3, 1, 1, activation='relu')
        self.scale3 = conv_block(n_out//4, n_out - n_out//4 - n_out//2, 3, 1, 1, activation=None)

        self.skip = conv_block(n_inp, n_out, 1, 1, 0, None) if n_inp != n_out else nn.Sequential()

    def forward(self, x):
        o1 = self.scale1(x)
        o2 = self.scale2(o1)
        o3 = self.scale3(o2)
        o4 = torch.cat([o1, o2, o3], 1)
        return o4 + self.skip(x)


class SoftArgmax2D(nn.Module):
    def __init__(self, beta=1):
        super(SoftArgmax2D, self).__init__()
        self.beta = beta

    def forward(self, hm):
        hm = hm.mul(self.beta)
        bs, nc, h, w = hm.size()
        hm = hm.squeeze()

        softmax = F.softmax(hm.view(bs, nc, h * w), dim=2).view(bs, nc, h, w)

        weights = torch.ones(bs, nc, h, w).float().to(hm.device)
        w_x = torch.arange(w).float().div(w)
        w_x = w_x.to(hm.device).mul(weights)

        w_y = torch.arange(h).float().div(h)
        w_y = w_y.to(hm.device).mul(weights.transpose(2, 3)).transpose(2, 3)

        approx_x = softmax.mul(w_x).view(bs, nc, h * w).sum(2).unsqueeze(2)
        approx_y = softmax.mul(w_y).view(bs, nc, h * w).sum(2).unsqueeze(2)

        res_xy = torch.cat([approx_x, approx_y], 2)
        return res_xy



class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs):
        b, n, d = inputs.shape
        n_s = self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots



class Hourglass(nn.Module):
    def __init__(self, n, hg_width, n_inp, n_out,upmode='nearest', verbose=False, layers=[]):
        super(Hourglass, self).__init__()

        self.layers = layers
        self.verbose = verbose
        self.n = n
        self.upmode = upmode

        self.lower0 = conv_block(n_inp, n_inp, ks=4,stride=2,padding=1,activation='relu', bias=False)
        self.lower1 = MultiScaleHGResidual(n_inp, hg_width)
        self.lower2 = MultiScaleHGResidual(hg_width, hg_width)
        self.lower3 = MultiScaleHGResidual(hg_width, hg_width)

        if n > 1:
            self.lower4 = Hourglass( n - 1, hg_width, hg_width, n_out, upmode, 
                                     verbose=self.verbose, layers=self.layers)
        else:
            self.lower4 = MultiScaleHGResidual(hg_width, n_out)

        self.lower5 = MultiScaleHGResidual(n_out, n_out)

        self.upper1 = MultiScaleHGResidual(n_inp, hg_width)
        self.upper2 = MultiScaleHGResidual(hg_width, hg_width)
        self.upper3 = MultiScaleHGResidual(hg_width, n_out)
    

    def forward(self, x):
        o0 = self.lower0(x)
        o1 = self.lower1(o0)
        o2 = self.lower2(o1)
        o3 = self.lower3(o2)

        if self.n >1:
            o4, layers = self.lower4(o3)
        else:
            o4 = self.lower4(o3)

        o5 = self.lower5(o4)
        self.layers.append(o5.detach())

        o1_u = self.upper1(x)
        o2_u = self.upper2(o1_u)
        o3_u = self.upper3(o2_u)

        if self.verbose:
            print()
            print(self.n)
            print(o1.shape)
            print(o4.shape)
            print(o1_u.shape)
            print('*******************************')
            print()

        return o3_u + F.interpolate(o5, x.size()[-2:], mode=self.upmode, align_corners=True), self.layers


def get_positional_encoding(dim_encoding, height, width, requires_grad=True, device='cuda:0'):
    # construction of positional encoding tables
    # first initialized with trigonometric encoding, then fine-tuned during the training phase
    positional_encoding = torch.zeros((1, dim_encoding, height, width))

    pi = 3.1415927410125732
    assert dim_encoding % 4 == 2
    positional_encoding = torch.zeros((1, dim_encoding, height, width))
    for x in range(width):
        for y in range(height):
            xn = (x / width - 0.5)
            yn = (y / height - 0.5)
            positional_encoding[0, 0 , y, x] = xn
            positional_encoding[0, 1, y, x] = yn
            for i in range((dim_encoding-2) // 4):
                xm = xn * (2 ** i)*pi
                ym = yn * (2 ** i)*pi
                positional_encoding[0, 2+0 + 4 * i, y, x] = math.sin(xm)
                positional_encoding[0, 2+1 + 4 * i, y, x] = math.sin(ym)
                positional_encoding[0, 2+2 + 4 * i, y, x] = math.cos(xm)
                positional_encoding[0, 2+3 + 4 * i, y, x] = math.cos(ym)

    print('created positional encoding table with ', height* width * dim_encoding, ' parameters')

    return nn.Parameter(positional_encoding, requires_grad=requires_grad).to(device)


