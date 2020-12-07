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
   
