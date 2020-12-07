import torch
from torch import nn
import torchvision.utils as vutils
from torch.autograd import Variable

import numpy as np

from Param import *
from utils import weights_init

from model import Final_Model

import gc

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def fit(data, model, optimizer, criterion, max_norm=0):

    img = Variable(data[0], volatile=False).to(device)

    output = model(img)

    loss = criterion(output, img)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss.detach_()
    del output

    if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    return loss


def train(dataloader, print_epoch=batch_size, verbose=False):

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    assert image_size == 128 or image_size == 256 or image_size == 512

    model = Final_Model(image_size, image_size, hidden= nf, n_heads = n_heads, nze=2, 
                         nc=nc, n_stack=n_stack, n_iters=n_iters, n_slots=n_slots)

    model = torch.nn.DataParallel(model)    
    model = model.to(device) 

    if initialize_weights:
        model.apply(weights_init)

    criterion = nn.MSELoss()  

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params ', n_parameters)


    print("Starting Training Loop...")


    losses = []
    img_list = []

    # For each epoch
    for epoch in range(num_epochs):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        gc.collect()
        model.train()

        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            gc.collect()

            if verbose: print(data[0].shape)
            if verbose: print(data[1].shape)

            recons_loss = fit(data, model, optimizer, criterion)

            print("Allocated: %fGB"%(torch.cuda.memory_allocated()/1024/1024/1024))
            print("Cached: %fGB"%(torch.cuda.memory_cached()/1024/1024/1024))
                

            # Output training stats
            if i % print_epoch == 0:
                print('[%d/%d][%d/%d]\tLoss: %.4f'
                        % (epoch+1, num_epochs, i, len(dataloader), recons_loss.item()))

            # Save Losses for plotting later
            losses.append(recons_loss.item())
            del recons_loss
            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    model.eval()
                    img_out = model(data[0].to(device))
                    img_out = img_out.detach().cpu()

                img_list.append(vutils.make_grid(img_out[0:10,0], nrow=5, normalize=True))
                del img_out
            torch.cuda.empty_cache()
        dataloader.reset()

    return losses, img_list, model





