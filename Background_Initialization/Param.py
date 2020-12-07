import torch



# Root directory for dataset
dataroot = '../../../input/Sequence/'

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 32

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size_H = 720
image_size_W = 1280

if image_size_H <= image_size_W:
    image_size = image_size_H
else:
    image_size = image_size_W

# Number of channels in the training images. For color images this is 3
nc = 3

# Number of filters 
nf = 8


# Number of training epochs
num_epochs = 50

# Learning rate for optimizers
lr = 0.0002

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
Tensor = torch.cuda.FloatTensor if (torch.cuda.is_available() and ngpu > 0) else torch.FloatTensor

save_outputModel = True




