import torch

from Prepare_Data import *

import cv2
from skimage import io
from skimage.transform import resize

from Param import *

from models.AE_Attention import AE_Attention

import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt

# Create the dataloader
dataloader1 = DataLoader(dataroot, image_size_H, image_size_W, 1, workers, Allow_Shuffle=False)
dataloader2 = DataLoader(dataroot2, image_size_H, image_size_W, 1, workers, Allow_Shuffle=False)


AE_model = AE_Attention().to(device)


model_path = '/home/simon/Desktop/Scene_Analysis/Results/Varna_datasets/AE/nz-50/'
model_name = 'AE_Attention512_batch32_epochs100_nz50-attendToFG'
out_path = 'Output_varna/'


AE_model.load_state_dict(torch.load(model_path + model_name + '.pth'))
AE_model.eval()


torch.manual_seed("0")
for i, (data1,data2) in enumerate(zip(dataloader1, dataloader2), 0):
    if i%1000 == 0:
        print('==================> ', i)
        img = data1[0]
        flow = data2[0]

        in_img= np.array(vutils.make_grid(img[0], padding=2, normalize=True))
        in_img = np.transpose(in_img, (1,2,0))

        io.imsave(out_path + 'in/' + str(i) +'.jpg', in_img)
    
        img = img.to(device)
        flow = flow.to(device)

        input_ = torch.cat((img,flow),1)

        output = AE_model(input_)

        output = output.cpu().detach()
        out_img= np.array(vutils.make_grid(output[0], padding=2, normalize=True))
        out_img = np.transpose(out_img, (1,2,0))
        print(out_img.shape, out_img.dtype)
        io.imsave(out_path + 'out/' + str(i) + '.jpg', out_img)




