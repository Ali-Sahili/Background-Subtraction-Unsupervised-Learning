import torch
from torch import nn
import torchvision.utils as vutils

import numpy as np

from Attention_On_Encoder.AE_Attention_Encoder import AE_Attention_Encoder


from Param import *
from utils import weights_init

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def fit(data, model, optimizer, criterion):

    img = data[0].to(device)
    
    output = model(img)

    loss = criterion(output, img)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss.detach_()

    return loss


def trainAOE(dataloader, print_epoch=batch_size, verbose=True):

    assert image_size == 512, "Input dimension is incompatible."

    model = AE_Attention_Encoder().to(device)
    
    if initialize_weights:
        model.apply(weights_init)

    if loss_ == True:
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()  

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    AE_losses = []
    img_list = []

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        model.train()
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
         
            if verbose: print(data[0].shape)
            if verbose: print(data[1].shape)

            recons_loss = fit(data, model, optimizer, criterion)

            # Output training stats
            if i % print_epoch == 0:
                print('[%d/%d][%d/%d]\tLoss_AE: %.4f'
                        % (epoch+1, num_epochs, i, len(dataloader), recons_loss.item()))

            # Save Losses for plotting later
            AE_losses.append(recons_loss.item())
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    img_out = model(data[0].to(device)).detach().cpu()

                img_list.append(vutils.make_grid(img_out[0:10], nrow=5, normalize=True))
            

    return AE_losses, img_list, model


