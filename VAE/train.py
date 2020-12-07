import torch
from torch.autograd import Variable
import torchvision.utils as vutils

from VAE.model import VAE, Encoder, Decoder, VAE256

import os
path = os.getcwd()
os.chdir("../")
from Param import *
os.chdir(path)


from VAE.Loss_functions import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def trainVAE(dataloader, print_epoch = 32, verbose = False):

    assert image_size == 512 or image_size == 256
    #

    torch.cuda.empty_cache()
    torch.manual_seed(10)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if image_size == 256:
        model = VAE256().to(device)
    elif image_size == 512:
        model = VAE().to(device)
        #Encoder_model = Encoder().to(device)
        #Decoder_model = Decoder().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #optimizer = torch.optim.Adam(list(Encoder_model.parameters())+list(Decoder_model.parameters()), lr=lr) 

    img_list = []
    losses = []
    
    mu_list = []
    logvar_list = []

    print("Starting Training Loop...")
    #Encoder_model.train()
    #Decoder_model.train()
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        #Encoder_model.train()
        #Decoder_model.train()
        model.train()

        for i, data in enumerate(dataloader, 0):
            img = Variable(data[0]).to(device)

            output, mu, logvar = model(img)

            #mu, logvar = Encoder_model(img)
           
            """
            if verbose: 
                mu_mean = torch.mean(mu)
                logvar_mean = torch.mean(logvar)

                print('means: ',mu_mean.item(), logvar_mean.item())

            std = logvar.mul(0.5).exp_()
            condition1 = (mu > (mu + 0.1*std))
            condition2 = (mu < (mu - 0.1*std)) 

            outliers = condition1 * condition2

            mu = mu[outliers == False] # = 0 # mu_mean
            logvar = logvar[outliers == False] #= 0 # logvar_mean

            if verbose:
                nb1 = mu.numel() - (mu == 0).sum()
                nb2 = logvar.numel() - (logvar == 0).sum()
                print('number of zeros: ', nb1.item(), nb2.item())
            """

            #output = Decoder_model(mu, logvar)
           

            loss = loss_function2(output, img, mu, logvar)
            losses.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss.detach_()

            mu_list.append(mu)
            logvar_list.append(logvar)

            if i % print_epoch == 0:
                print('[%d/%d][%d/%d]\tLoss: %.4f'
                        % (epoch+1, num_epochs, i, len(dataloader), loss.item()))

            # Check how the generator is doing by saving G's output on fixed_noise
            if (epoch == num_epochs-1) and (i == len(dataloader)-1):
                img_list.append(vutils.make_grid(output.detach().cpu()[:8], padding=2, normalize=True))


    return img_list, losses, model, mu_list, logvar_list
