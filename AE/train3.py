import torch
from torch import nn
import torchvision.utils as vutils
import torchvision.transforms as transforms

import numpy as np

from models.Encoders import *
from models.Decoders import *
from models.patchmodels import Encoder512_F, Decoder512_F
from models.patchmodels_Attention import Encoder_Decoder512
from MultiScale_input.model import FinalModel

from Param import *
from utils import Covariance_Correlation, weights_init, add_sn

from Losses import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def toGray(input):
        final_output = None
        batch_size, channels, h, w = input.shape
        input_ = torch.squeeze(input, 1)

        for img in input_:
            img_PIL = transforms.ToPILImage()(img)
            img_PIL = img_PIL.convert('L')
            img_PIL = transforms.ToTensor()(img_PIL)
            #print(img_PIL.shape)
            if final_output is None:
                final_output = img_PIL
            else:
                final_output = torch.cat((final_output, img_PIL), 0)

        final_output = torch.unsqueeze(final_output, 1)
        return final_output.view(batch_size, 1, h, w)



def fit(data, Encoder, Decoder, optimizer, criterion):

    img = data[0]

    gray = toGray(img)

    input_ = torch.cat((img,gray),1).to(device)

    #print(input_.shape)
    #assert(0)
    encod_out = Encoder(input_)
    output = Decoder(encod_out)

    loss = criterion(input_, output)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss.detach_()

    return loss



def trainAE_Gray(dataloader, print_epoch=batch_size, verbose=True):

    
    if image_size_W == image_size_H:
        image_size = image_size_H

    if image_size == 256:
                Encoder_model = Encoder256(nc=4).to(device)
                Decoder_model = Decoder256(nc=4).to(device)
    elif image_size == 512:
                Encoder_model = Encoder512(nc=4).to(device)
                Decoder_model = Decoder512(nc=4).to(device)
    else:
        assert(0)
    
    if initialize_weights:
            print('spectral normalization ...')
            Encoder_model.apply(weights_init)
            Encoder_model.apply(add_sn)
            Decoder_model.apply(weights_init)
            Decoder_model.apply(add_sn)

    if loss_ == True:
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()  

    optimizer = torch.optim.Adam(list(Encoder_model.parameters())+list(Decoder_model.parameters()), lr=lr)

    print("Starting Training Loop...")


    AE_losses = []
    img_list = []
    # For each epoch
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        Encoder_model.train()
        Decoder_model.train()
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
         
            if verbose: print(data[0].shape)
            if verbose: print(data[1].shape)

            recons_loss = fit(data, Encoder_model, Decoder_model, optimizer, criterion)

            # Output training stats
            if i % print_epoch == 0:
                print('[%d/%d][%d/%d]\tLoss_AE: %.4f'
                        % (epoch+1, num_epochs, i, len(dataloader),
                           recons_loss.item()))

            # Save Losses for plotting later
            AE_losses.append(recons_loss.item())
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    enc_out = Encoder_model(torch.cat((data[0],toGray(data[0])),1).to(device))

                    if verbose: print('latent space: ',enc_out.shape)
                    img_out = Decoder_model(enc_out)[:,0:3,:,:].detach().cpu()
                    #print(img_out.shape)
                img_list.append(vutils.make_grid(img_out[0:10], nrow=5, normalize=True))
            

    return AE_losses, img_list, Encoder_model, Decoder_model




