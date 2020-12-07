import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

import torchvision.utils as vutils

from Param import *
from AAE.models import *


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

def imq_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
    batch_size = X.size(0)

    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size)
        stats += res1 - res2

    return stats


def rbf_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
    batch_size = X.size(0)

    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 / scale
        res1 = torch.exp(-C * dists_x)
        res1 += torch.exp(-C * dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = torch.exp(-C * dists_c)
        res2 = res2.sum() * 2. / batch_size
        stats += res1 - res2

    return stats

def trainWAE_MMD(dataloader, print_epoch=batch_size, verbose=True):

    torch.manual_seed(123)
    torch.cuda.empty_cache()

    if image_size == 64:
        encoder, decoder = Encoder64(), Decoder64()
    elif image_size == 128:
        encoder, decoder = Encoder128(), Decoder128()
    elif image_size == 256:
        encoder, decoder = Encoder256(), Decoder256()
    elif image_size == 512:
        encoder, decoder = Encoder512(), Decoder512()
    else:
        assert(0)

    criterion = nn.MSELoss()

    encoder.train()
    decoder.train()


    encoder, decoder = encoder.to(device), decoder.to(device)

    one = torch.Tensor([1])
    mone = one * -1

    one = one.to(device)
    mone = mone.to(device)

    # Optimizers
    enc_optim = optim.Adam(encoder.parameters(), lr=lr)
    dec_optim = optim.Adam(decoder.parameters(), lr=lr)

    enc_scheduler = StepLR(enc_optim, step_size=30, gamma=0.5)
    dec_scheduler = StepLR(dec_optim, step_size=30, gamma=0.5)

    sigma = 1
    LAMBDA = 10.0


    print("Starting Training Loop...")

    AE_losses = []
    img_list = []

    # For each epoch
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()

        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            images = data[0].to(device)

            enc_optim.zero_grad()
            dec_optim.zero_grad()

            # ======== Train Generator ======== #

            z = encoder(images)
            x_recon = decoder(z)

            recon_loss = criterion(x_recon, images)

            # ======== MMD Kernel Loss ======== #

            z_fake = Variable(torch.randn(batch_size, nz) * sigma)
            z_fake = z_fake.to(device)

            z_real = encoder(images)

            mmd_loss = imq_kernel(z_real, z_fake, h_dim=nz)
            mmd_loss = mmd_loss / batch_size

            total_loss = recon_loss + mmd_loss
            total_loss.backward()

            enc_optim.step()
            dec_optim.step()

            AE_losses.append(recon_loss.item())

            # Output training stats
            if i % print_epoch == 0:
                print('[%d/%d][%d/%d]\tRecons Loss: %.4f, MMD Loss %.4f'
                        % (epoch+1, num_epochs, i, len(dataloader), recon_loss.item(), 
                                                                   mmd_loss.item()))
            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    img_out = x_recon.detach().cpu()
                img_list.append(vutils.make_grid(img_out[0:10], nrow=5, normalize=True))

    return AE_losses, img_list, encoder, decoder


