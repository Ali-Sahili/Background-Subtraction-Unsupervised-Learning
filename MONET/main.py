from Prepare_Data import DataLoader

from train import train

from Param import *
from utils import *

import time


print()
# Create the dataloader
dataloader = DataLoader(dataroot, image_size, image_size, batch_size, workers, Allow_Shuffle=Allow_Shuffle)
print('Number of images used in training: ', len(dataloader)*batch_size)


print()
print('Input image size: ', image_size)
#print('Latent space dimension: ', nz)
print('Batch size: ', batch_size)
print('Number of epochs: ', num_epochs)
print()

start_time = time.process_time()

losses, img_list, model = train(dataloader, print_epoch=1)

print("\nProcessing time = ", time.process_time()-start_time, " s")


if save_outputModel:
    model_name = "model_"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+ ".pth"
    save_model(model, model_name)
    
    AE_losses_plot(losses)

    Results_plot(dataloader, img_list)
#####################################################################

