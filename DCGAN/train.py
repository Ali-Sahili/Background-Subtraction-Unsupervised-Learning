import torch
import torch.nn as nn

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim

import torchvision.utils as vutils

from DCGAN.models import *

from Param import nz, lr, num_epochs, batch_size, image_size
from Param import device, ngpu, Unet, SN_ind, initialize_weights

from utils import weights_init

def fit(data, netG, netD, criterion, optimizerG, optimizerD, real_label, fake_label):

    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
        
    ## Train with all-real batch
    netD.zero_grad()
    
    # Format batch
    real_data = data[0].to(device)
    b_size = real_data.size(0)
    label = torch.full((b_size,), real_label, device=device)

    # Forward pass real batch through D
    output = netD(real_data).view(-1)
    #print('real: ',real_data.shape, ' out: ',output.shape)
    
    # Calculate loss on all-real batch
    #criterion2 = nn.MSELoss()
    errD_real = criterion(output, label) #+ criterion2(output, label)
    
    # Calculate gradients for D in backward pass
    errD_real.backward()
    #D_x = output.mean().item()

    ## Train with all-fake batch
    # Generate batch of latent vectors
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    #print('noise: ',noise.shape)    
    
    # Generate fake image batch with G
    fake = netG(noise)
    #label = torch.full((fake.size(0),), fake_label, device=device)
    label.fill_(fake_label)

    # Classify all fake batch with D
    output = netD(fake.detach()).view(-1)
    #print('fake: ',fake.shape, ' out: ',output.shape)

    # Calculate D's loss on the all-fake batch
    errD_fake = criterion(output, label)
    
    # Calculate the gradients for this batch
    errD_fake.backward()
    #D_G_z1 = output.mean().item()
    
    # Add the gradients from the all-real and all-fake batches
    errD = errD_real + errD_fake
  
    # Update D
    optimizerD.step()

    
    ############################    
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    label.fill_(real_label)  # fake labels are real for generator cost
        
    # Since we just updated D, perform another forward pass of all-fake batch through D
    output = netD(fake).view(-1)
        
    # Calculate G's loss based on this output
    errG = criterion(output, label)
    
    # Calculate gradients for G
    errG.backward()
    #D_G_z2 = output.mean().item()
    
    # Update G
    optimizerG.step()

    errG.detach_()
    errD.detach_()

    return errG, errD


def trainDCGAN(dataloader, print_epoch=1, verbose=True):

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
        if SN_ind: netD = Discriminator512_SN().to(device)
        else: netD = Discriminator512().to(device)
    elif image_size == 512 and Unet == True:
        netG = Generator512().to(device)
        netD = netD512().to(device)
    

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    if initialize_weights and not SN_ind:
        netG.apply(weights_init)
        netD.apply(weights_init)

    # Print the model
    if verbose: print(netG)
    if verbose: print(netD)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    #beta1 = 0.5
    #optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    #optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=0.0005)
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

            errG, errD = fit(data, netG, netD, criterion, optimizerG,
                                                    optimizerD, real_label, fake_label)

            # Output training stats
            if i % print_epoch == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                        % (epoch, num_epochs, i, len(dataloader),
                           errD.item(), errG.item()))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    return G_losses, D_losses, img_list, netG



