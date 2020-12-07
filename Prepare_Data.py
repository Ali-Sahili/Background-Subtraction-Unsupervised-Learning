import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

# We can use an image folder dataset the way we have it setup.
# Create the dataset
def DataLoader(dataroot, image_size_H, image_size_W, batch_size, workers, Allow_Shuffle=False):
    print('Loading training data ...')
    dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize((image_size_H,image_size_W)),
                               #transforms.CenterCrop((image_size_H+image_size_W)/2),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    # Create the dataloader
    if not Allow_Shuffle:
        sampler = torch.utils.data.SequentialSampler(dataset)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=Allow_Shuffle, num_workers=workers, 
                                                  sampler= sampler, 
                                                  drop_last=True)

    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=Allow_Shuffle, num_workers=workers, 
                                                  drop_last=True)

    return dataloader
    


""" Read optical flow dataset as numpy array of dimensions H x W x 2  """
def DataLoader_flows(dataroot2, image_size_H, image_size_W, batch_size, workers, Allow_Shuffle=False):
    import numpy as np
    def npy_loader(path):
        """ read .npy file and convert it to tensor type
            to apply transformation, you should convert 
               the sample tensors to PIL.images   """
        sample = torch.from_numpy(np.load(path))
        return sample

    print('Loading training flows ...')
    dataset = dset.DatasetFolder(root = dataroot2,
                                 loader = npy_loader,
                                 extensions = ['.npy'],
                           transform=transforms.Compose([
                               transforms.ToPILImage(),
                               transforms.Resize((image_size_H,image_size_W)),
                               #transforms.CenterCrop((image_size_H+image_size_W)/2),
                               transforms.ToTensor(),
                               #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    # Create the dataloader
    sampler = torch.utils.data.SequentialSampler(dataset)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=Allow_Shuffle, num_workers=workers, sampler= sampler, 
                                                  drop_last=True)

    return dataloader
