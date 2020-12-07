import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import pytorch_ssim

import torchvision.utils as vutils

import numpy as np

from AAE.models import *

from Param import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



# Train procedure for one epoch
def fit(data, P, Q, D_gauss, P_decoder, Q_encoder, Q_generator, D_gauss_solver, criterion):

    imgs = data[0].to(device)


    # Init gradients
    P.zero_grad()
    Q.zero_grad()
    D_gauss.zero_grad()

    #----------------------
    # Reconstruction phase
    
    #----------------------    
    z_sample = Q(imgs)
    output = P(z_sample)

    recon_loss = criterion(output, imgs)       
    #recon_loss_2 = 1 - pytorch_ssim.SSIM()(output, imgs)

    #k = 0.2
    #recon_loss = (1-k)*recon_loss_1 + k*recon_loss_2

    recon_loss.backward()
    P_decoder.step()
    Q_encoder.step()

    P.zero_grad()
    Q.zero_grad()
    D_gauss.zero_grad()

    #----------------------
    # Regularization phase
    #----------------------
    # Discriminator
    Q.eval()
    z_real_gauss = Variable(torch.randn(batch_size, nz) * 5.).to(device)

    z_fake_gauss = Q(imgs)

    D_real_gauss = D_gauss(z_real_gauss)
    D_fake_gauss = D_gauss(z_fake_gauss)

    D_loss = -torch.mean(torch.log(D_real_gauss + TINY) + torch.log(1 - D_fake_gauss + TINY))

    D_loss.backward()
    D_gauss_solver.step()

    P.zero_grad()
    Q.zero_grad()
    D_gauss.zero_grad()

    # Generator
    Q.train()
    z_fake_gauss = Q(imgs)

    D_fake_gauss = D_gauss(z_fake_gauss)
    G_loss = -torch.mean(torch.log(D_fake_gauss + TINY))

    G_loss.backward()
    Q_generator.step()

    P.zero_grad()
    Q.zero_grad()
    D_gauss.zero_grad()

    G_loss.detach_()
    D_loss.detach_()
    recon_loss.detach_()

    del z_fake_gauss, D_real_gauss, D_fake_gauss 

    return D_loss, G_loss, recon_loss

""" For Wasserstein AAE """
def fit2(data, P, Q, D, P_solver, Q_solver, D_solver, criterion):
        X = Variable(data[0]).to(device)


        z_sample = Q(X)

        X_sample = P(z_sample)
        recon_loss = criterion(X_sample, X)

        recon_loss.backward()
        P_solver.step()
        Q_solver.step()

        Q.zero_grad()
        P.zero_grad()
        D.zero_grad()

        """ Regularization phase """
        # Discriminator
        for _ in range(5):
            z_real = Variable(torch.randn(batch_size, nz)).to(device)

            z_fake = Q(X).view(batch_size,-1)

            D_real = D(z_real)
            D_fake = D(z_fake)

            #D_loss = -torch.mean(torch.log(D_real) + torch.log(1 - D_fake))
            D_loss = -(torch.mean(D_real) - torch.mean(D_fake))

            D_loss.backward()
            D_solver.step()

            # Weight clipping
            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

            Q.zero_grad()
            P.zero_grad()
            D.zero_grad()

            D_loss.detach_()

        # Generator
        z_fake = Q(X).view(batch_size,-1)
        D_fake = D(z_fake)



        #G_loss = -torch.mean(torch.log(D_fake))
        G_loss = -torch.mean(D_fake)

        G_loss.backward()
        Q_solver.step()
        Q.zero_grad()
        P.zero_grad()
        D.zero_grad()

        D_loss.detach_()
        G_loss.detach_()
        recon_loss.detach_()

        del z_fake, D_real, D_fake

        return D_loss, G_loss, recon_loss



def trainAAE(dataloader, print_epoch=32, verbose=False):

    torch.manual_seed(10)
    torch.cuda.empty_cache()

    # Create encoder and decoder models
    if image_size == 64:
        Encoder_model = Encoder64().to(device)
        Decoder_model = Decoder64().to(device)
    elif image_size == 128:
        Encoder_model = Encoder128().to(device)
        Decoder_model = Decoder128().to(device)
    elif image_size == 256:
        Encoder_model = Encoder256().to(device)
        Decoder_model = Decoder256().to(device)
    elif image_size == 512:
        Encoder_model = Encoder512().to(device)
        Decoder_model = Decoder512().to(device)


    D_gauss = Discriminator().to(device)

    # Set optimizators
    P_decoder = optim.Adam(Decoder_model.parameters(), lr=gen_lr)
    Q_encoder = optim.Adam(Encoder_model.parameters(), lr=gen_lr)# to have independence in the optimization 
                                                     # procedure for the encoder, we take 2 optimizers
    Q_generator = optim.Adam(Encoder_model.parameters(), lr=reg_lr)
    D_gauss_solver = optim.Adam(D_gauss.parameters(), lr=reg_lr)

    if loss_:
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss() 
        #criterion = nn.SmoothL1Loss()

    print("Starting Training Loop...")

    iters = 0
    AAE_losses = []
    G_losses = []
    D_losses = []
    img_list = []


    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        # Set the networks in train mode (apply dropout when needed)
        Encoder_model.train()
        Decoder_model.train()
        D_gauss.train()
        
        for i, data in enumerate(dataloader, 0):

            if verbose: print(data[0].shape)
            if verbose: print(data[1].shape)

            if method == 'WAAE': D_loss_gauss, G_loss, recon_loss = fit2(data, Decoder_model, Encoder_model, D_gauss, P_decoder, Q_encoder, D_gauss_solver, criterion)

            else: D_loss_gauss, G_loss, recon_loss = fit(data, Decoder_model, Encoder_model, D_gauss, P_decoder, Q_encoder, Q_generator, D_gauss_solver, criterion)

            # Output training stats
            if i % print_epoch == 0:
                print('[%d/%d][%d/%d]\tLoss_AAE: %.4f\tD_Loss: %.8f'
                        % (epoch+1, num_epochs, i, len(dataloader),
                           recon_loss.item(), D_loss_gauss.item()))

            # Save Losses for plotting later
            AAE_losses.append(recon_loss.item())
            G_losses.append(G_loss.item())
            D_losses.append(D_loss_gauss.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    enc_out = Encoder_model(data[0].to(device))
                    if verbose: print('latent space: ',enc_out.shape)
                    img_out = Decoder_model(enc_out).detach().cpu()
                img_list.append(vutils.make_grid(img_out, padding=2, normalize=True))
            
            iters += 1

    return AAE_losses, G_losses, D_losses, img_list, Encoder_model, Decoder_model














