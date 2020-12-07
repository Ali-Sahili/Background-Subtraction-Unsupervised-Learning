import torch
import torchvision.utils as vutils
from Prepare_Data import DataLoader

from model import BENet, Initialize, Initialize2

from Param import *
from utils import *

import time

torch.backends.cudnn.deterministic = True
# custom weights initialization called on netG and netE
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(1.0, 0.02)

# Create the dataloader
dataloader = DataLoader(dataroot, image_size_H, image_size_W, batch_size, workers, Allow_Shuffle=False)

start_time = time.process_time()

BENet_model = Initialize().to(device)
#BENet_model.apply(weights_init)

BG_list = []
FG_list = []
print("Starting Loop...")
for i, data in enumerate(dataloader, 0):

    img = data[0].to(device)
    background, foreground = BENet_model(img)
    fg = foreground.detach().cpu()
    bg = background.detach().cpu()

    BG_list.append(vutils.make_grid(bg[0:10], nrow=5, normalize=True))
    FG_list.append(vutils.make_grid(fg[0:10], nrow=5, normalize=True))

print("\nProcessing time = ", time.process_time()-start_time, " s")


Results_plot(dataloader, BG_list, FG_list)










