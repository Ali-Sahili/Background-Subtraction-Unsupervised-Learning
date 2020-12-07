import torch
from torch import nn
from torch.autograd import Variable
import torchvision.utils as vutils

import numpy as np

from models.Encoders import Encoder512, Encoder256
from models.Decoders import Decoder512, Decoder256
from AE_GMMN.GMMN import GMMN

from Param import batch_size, image_size, nz, device, num_epochs, lr
from utils import Covariance_Correlation, weights_init

from Losses import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

def get_scale_matrix(M, N):
    s1 = (torch.ones((N, 1)) * 1.0 / N).to(device)
    s2 = (torch.ones((M, 1)) * -1.0 / M).to(device)
    return torch.cat((s1, s2), 0)

def train_one_step(latent_vector, samples, model, gmmn_optimizer, sigma=[1]):

    samples = Variable(samples).to(device)
    gen_samples = model(samples)

    latent_vector = latent_vector.view(-1, nz)
    gen_samples = gen_samples.view(-1, nz)

    X = torch.cat((gen_samples, latent_vector), 0)
    XX = torch.matmul(X, X.t())
    X2 = torch.sum(X * X, 1, keepdim=True)
    exp = XX - 0.5 * X2 - 0.5 * X2.t()

    M = gen_samples.shape[0]
    N = latent_vector.shape[0]
    s = get_scale_matrix(M, N)
    S = torch.matmul(s, s.t())

    loss = 0
    for v in sigma:
        kernel_val = torch.exp(exp / v)
        loss += torch.sum(S * kernel_val)

    loss = torch.sqrt(loss)

    gmmn_optimizer.zero_grad()
    loss.backward()
    gmmn_optimizer.step()

    return loss, gen_samples

def trainAE_GMMN(dataloader, print_epoch=batch_size, verbose=True):

    
    assert image_size == 512 or image_size == 256

    if image_size == 256:
        AE_Encoder = Encoder256().to(device)
        AE_Decoder = Decoder256().to(device)
    elif image_size == 512:
        AE_Encoder = Encoder512().to(device)
        AE_Decoder = Decoder512().to(device)

    GMMN_model = GMMN(nz, nz).to(device)


    AE_Encoder.apply(weights_init)
    AE_Decoder.apply(weights_init)
    GMMN_model.apply(weights_init)

    lr_gmmn = 0.001
    optimizer_gmmn = torch.optim.Adam(GMMN_model.parameters(), lr=lr_gmmn)
    optimizer = torch.optim.Adam(list(AE_Encoder.parameters())+list(AE_Decoder.parameters()), lr=lr)

    criterion = nn.MSELoss()

    print("Starting Training Loop...")


    AE_losses = []
    img_list = []

    encod_out = torch.zeros((batch_size, nz))

    # For each epoch
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        GMMN_model.train()

        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):


            # ================     AE train =================
            free_params(AE_Encoder)
            free_params(AE_Decoder)
            frozen_params(GMMN_model)
            
            imgs = Variable(data[0]).to(device)
            encod_out = AE_Encoder(imgs)
            new_encod_out = GMMN_model(encod_out)
            output = AE_Decoder(new_encod_out)

            loss = criterion(output, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ============ GMMN model train =============
            frozen_params(AE_Encoder)
            frozen_params(AE_Decoder)
            free_params(GMMN_model)

            # uniform random noise between [-1, 1]
            random_noise = torch.rand((batch_size, nz)) * 2 - 1
            loss, gen_samples = train_one_step(encod_out.detach(), random_noise, GMMN_model, optimizer_gmmn)

            # Output training stats
            if i % print_epoch == 0:
                print('[%d/%d][%d/%d]\tLoss_AE: %.4f'
                        % (epoch+1, num_epochs, i, len(dataloader), loss.item()))

            # Save Losses for plotting later
            AE_losses.append(loss.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    gen_samples = gen_samples.view(encod_out.shape).detach()
                    img_out = AE_Decoder(gen_samples)
                    img_out = img_out.detach().cpu()
                img_list.append(vutils.make_grid(img_out[:10],nrow=5, normalize=True))
            

    return AE_losses, img_list, GMMN_model, AE_Encoder, AE_Decoder




