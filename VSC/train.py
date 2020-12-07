import torch
from torch.autograd import Variable
import torchvision.utils as vutils

from VSC.model import *

import os
path = os.getcwd()
os.chdir("../")
from Param import *
os.chdir(path)


from VSC.Loss_functions import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def trainVSC(dataloader, print_epoch = 32, verbose = False):

    assert image_size == 512 or image_size == 256
    #model = VSC512().to(device)

    #Encoder_model = VSC_Encoder().to(device)
    #Decoder_model = VSC_Decoder().to(device)

    if image_size == 256:
        model = VSC256().to(device)
    elif image_size == 512:
        model = VSC512().to(device)
    else:
        assert(0)
 
    # Tune the learning rate (All training rates used were between 0.001 and 0.01)
    lr_vsc = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #optimizer = torch.optim.Adam(list(Encoder_model.parameters())+list(Decoder_model.parameters()), lr=lr_vsc) 

    img_list = []
    losses = []
    recon_losses = []
    #mu_list = []
    #logvar_list = []

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

            optimizer.zero_grad()

            output, mu, logvar, logspike = model(img)

            #mu, logvar, logspike = Encoder_model(img)

            #output = Decoder_model(mu, logvar, logspike)
           

            loss, recon_loss = loss_function(output, img, mu, logvar, logspike)
            losses.append(loss)
            recon_losses.append(recon_loss)

            loss.backward()
            optimizer.step()


            #mu_list.append(mu)
            #logvar_list.append(logvar)

            if i % print_epoch == 0:
                print('[%d/%d][%d/%d]\tLoss: %.4f, recon_loss %.4f'
                        % (epoch+1, num_epochs, i, len(dataloader), loss.item(), recon_loss.item()))

            # Check how the generator is doing by saving G's output on fixed_noise
            if (epoch == num_epochs-1) and (i == len(dataloader)-1):
                img_list.append(vutils.make_grid(output.detach().cpu()[:8], padding=2, normalize=True))


    return img_list, losses, recon_losses, model #Encoder_model, Decoder_model
