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

def trainWAE_GAN(dataloader, print_epoch=batch_size, verbose=True):

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

    discriminator = Discriminator()
    criterion = nn.MSELoss()

    encoder.train()
    decoder.train()
    discriminator.train()

    # Optimizers
    enc_optim = optim.Adam(encoder.parameters(), lr = lr)
    dec_optim = optim.Adam(decoder.parameters(), lr = lr)
    dis_optim = optim.Adam(discriminator.parameters(), lr = 0.5 * lr)

    enc_scheduler = StepLR(enc_optim, step_size=30, gamma=0.5)
    dec_scheduler = StepLR(dec_optim, step_size=30, gamma=0.5)
    dis_scheduler = StepLR(dis_optim, step_size=30, gamma=0.5)


    encoder, decoder = encoder.to(device), decoder.to(device)
    discriminator = discriminator.to(device)

    one = torch.Tensor([1])
    mone = one * -1

    one = one.to(device)
    mone = mone.to(device)

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

            encoder.zero_grad()
            decoder.zero_grad()
            discriminator.zero_grad()

            # ======== Train Discriminator ======== #

            frozen_params(decoder)
            frozen_params(encoder)
            free_params(discriminator)

            z_fake = torch.randn(batch_size, nz) * sigma
            z_fake = z_fake.to(device)

            d_fake = discriminator(z_fake)

            z_real = encoder(images)
            d_real = discriminator(z_real)

            torch.log(d_fake).mean().backward(mone)
            torch.log(1 - d_real).mean().backward(mone)

            dis_optim.step()

            # ======== Train Generator ======== #

            free_params(decoder)
            free_params(encoder)
            frozen_params(discriminator)


            z_real = encoder(images)
            x_recon = decoder(z_real)
            d_real = discriminator(encoder(Variable(images)))

            recon_loss = criterion(x_recon, images)
            d_loss = LAMBDA * (torch.log(d_real)).mean()

            recon_loss.backward(one)
            d_loss.backward(mone)

            enc_optim.step()
            dec_optim.step()

            # Save Losses for plotting later
            AE_losses.append(recon_loss.item())

            # Output training stats
            if i % print_epoch == 0:
                print('[%d/%d][%d/%d]\tLoss_AE: %.4f'
                        % (epoch+1, num_epochs, i, len(dataloader), recon_loss.item()))

            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    img_out = x_recon.detach().cpu()
                img_list.append(vutils.make_grid(img_out[0:10], nrow=5, normalize=True))

    return AE_losses, img_list, encoder, decoder



