import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from Param import dataroot, dataroot2
from Prepare_Data import DataLoader, DataLoader_flows

import numpy as np
import matplotlib.pyplot as plt



def Plot(img):
	plt.figure(figsize=(10,10))
	plt.axis("off")
	plt.title("Real Images")
	plt.imshow(np.transpose(vutils.make_grid(img[:10], nrow=5, normalize=True), (1,2,0)))
	plt.show()

    
def test_data_imgs():
    dataloader = DataLoader(dataroot, 512, 512, 32, 2, False)

    for i, data in enumerate(dataloader):
        img = data[0] 
        print(i)
        Plot(img)


def test_data_flows(path):
    import cv2
    dataloader = DataLoader_flows(path, 512, 512, 32, 2, False)

    for i, data in enumerate(dataloader):
        print(i)
        flow_map = data[0]
        print(flow_map.shape, flow_map.dtype)
        flow = flow_map[0].permute(2,1,0)
        print(flow.shape)
        img = flow.numpy()
        cv2.imshow('img', img)
        cv2.waitKey(0)
        if i == 2:
            break

path = '../../../Varna_datasets/Flows_npy/'
test_data_flows(path)
