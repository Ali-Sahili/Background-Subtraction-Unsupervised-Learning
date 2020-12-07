import torch
import torch.nn as nn

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim

from torch.autograd import Variable

import torchvision.utils as vutils

from DCGAN.models import *

from Param import nz, lr, num_epochs, batch_size, image_size
from Param import device, ngpu, Unet, Tensor

from utils import weights_init

import numpy as np




def trainWDCGAN(dataloader, print_epoch=1, verbose=True):

    torch.cuda.empty_cache()
    torch.manual_seed(10)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


    # Create the generator
    if image_size == 64:
        netG = Generator64().to(device)
        netD = Discriminator64().to(device)
    elif image_size == 128:
        netG = Generator128().to(device)
        netD = Discriminator128().to(device)
    elif image_size == 256:
        netG = Generator256().to(device)
        netD = Discriminator256().to(device)
    elif image_size == 512 and Unet == False:
        netG = Generator512().to(device)
        netD = Discriminator512().to(device)
    elif image_size == 512 and Unet == True:
        netG = Generator512().to(device)
        netD = netD512().to(device)
    

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Print the model
    if verbose: print(netG)
    if verbose: print(netD)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)


    # Setup Adam optimizers for both G and D
    #beta1 = 0.5
    #optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    #optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=0.00005)
    optimizerD = torch.optim.RMSprop(netD.parameters(), lr=0.00005)

    # Training Loop
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")

    # For each epoch
    for epoch in range(num_epochs):
        netG.train()
        netD.train()
        torch.cuda.empty_cache()
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            if verbose: print(data[0].shape)
            if verbose: print(data[1].shape)

                # Configure input
            real_imgs = Variable(data[0]).to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizerD.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (data[0].shape[0], nz)))).to(device)

            # Generate a batch of images
            fake_imgs = netG(z).detach()
            # Adversarial loss
            loss_D = -torch.mean(netD(real_imgs)) + torch.mean(netD(fake_imgs))

            loss_D.backward()
            optimizerD.step()
            loss_D.detach_()

            # Clip weights of discriminator
            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01)

            # Train the generator every n_critic iterations
            if i % 5 == 0:

                # -----------------
                #  Train Generator
                # -----------------

                optimizerG.zero_grad()

                # Generate a batch of images
                gen_imgs = netG(z)
                # Adversarial loss
                loss_G = -torch.mean(netD(gen_imgs))

                loss_G.backward()
                optimizerG.step()

                loss_G.detach_()

                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                        % (epoch, num_epochs, i, len(dataloader),
                           loss_D.item(), loss_G.item()))

                # Save Losses for plotting later
                G_losses.append(loss_G.item())
                D_losses.append(loss_D.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                vutils.save_image(fake.data[:25], "output_varna/%d.png" % i, nrow=5, normalize=True)

            iters += 1

    return G_losses, D_losses, img_list, netG



