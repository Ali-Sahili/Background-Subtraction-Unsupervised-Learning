import torch
from torch import nn
from torch.autograd import Variable

import torchvision.utils as vutils
import torchvision.transforms as transforms

import numpy as np

from OneEncoder_MultiDecoders.models import Model512, VAE_Model512
from VAE.Loss_functions import *

from Param import *
from utils import weights_init

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize_tensor(input, size):
        final_output = None
        batch_size, channels, h, w = input.shape
        input_ = torch.squeeze(input, 1)

        for img in input_:
            img_PIL = transforms.ToPILImage()(img)
            img_PIL = transforms.Resize((size,size))(img_PIL)
            img_PIL = transforms.ToTensor()(img_PIL)
            if final_output is None:
                final_output = img_PIL
            else:
                final_output = torch.cat((final_output, img_PIL), 0)

        final_output = torch.unsqueeze(final_output, 1)
        return final_output.view(batch_size, channels, size, size)



def fit(data, model, optimizer, criterion):

    img = data[0].to(device)
      
    img128 = resize_tensor(img.cpu(), size=128).detach().to(device)
    img256 = resize_tensor(img.cpu(), size=256).detach().to(device)

    if method == 'OneEncoder_MultiDecoders':
        out128, out256, out512 = model(img)

        loss128 = criterion(out128, img128)
        loss256 = criterion(out256, img256)    
        loss512 = criterion(out512, img)

    elif method == 'OneEncoder_MultiDecoders_VAE':
        out128, out256, out512, mu, logvar = model(img)

        loss128 = loss_function2(out128, img128, mu, logvar)
        loss256 = loss_function2(out256, img256, mu, logvar)    
        loss512 = loss_function2(out512, img, mu, logvar)

    loss = loss128 + loss256 + loss512

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss.detach_()

    del loss128, loss256, loss512
    del out128, out256

    return loss, out512



def trainOEMD(dataloader, print_epoch=batch_size, verbose=True):

    
    if image_size_W == image_size_H:
        image_size = image_size_H


    assert image_size == 512



    if method == 'OneEncoder_MultiDecoders': model = Model512()
    elif method =='OneEncoder_MultiDecoders_VAE': model = VAE_Model512()
    else: assert(0)

    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr, weight_decay=1e-5)
    model = model.to(device)

    if initialize_weights:
        model.apply(weights_init)

    if loss_:
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()  



    print("Starting Training Loop...")
    AE_losses = []
    img_list = []


    # For each epoch
    for epoch in range(num_epochs):
        model.train()
        torch.cuda.empty_cache()
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
         
            if verbose: print(data[0].shape)
            if verbose: print(data[1].shape)


            recons_loss, img_out = fit(data, model, optimizer, criterion)

            # Output training stats
            if i % print_epoch == 0:
                print('[%d/%d][%d/%d]\tLoss_AE: %.4f'
                        % (epoch+1, num_epochs, i, len(dataloader),
                           recons_loss.item()))

            # Save Losses for plotting later
            AE_losses.append(recons_loss.item())
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):

                img_out = img_out.detach().cpu()
                img_list.append(vutils.make_grid(img_out[0:10], nrow=5, normalize=True))


    return AE_losses, img_list, model



