import os
import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import cv2
import numpy as np

# We can use an image folder dataset the way we have it setup.
# Create the dataset
def DataLoader(dataroot, image_size, batch_size, workers):
    print('Loading training data ...')
    dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize((image_size,image_size)),
                               transforms.ToTensor(),
                           ]))
    # Create the dataloader
    sampler = torch.utils.data.SequentialSampler(dataset)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=workers, 
                                                  sampler= sampler, 
                                                  drop_last=True)


    return dataloader




def confusionMatrix_compare(imBinary, imGT):
    """ Compares a binary frames with the groundtruth frame """
    v = 85
    TP = np.sum((imGT==255) * (imBinary==255))		# True Positive 
    TN = np.sum((imGT<=v) * (imBinary==0))		# True Negative
    FP = np.sum((imGT<=v) * (imBinary==255))    	        # False Positive
    FN = np.sum((imGT==255) * (imBinary==0))		# False Negative
    SE = np.sum((imGT==v) * (imBinary==255))		# Shadow Error
	
    p = imGT.shape[0] * imGT.shape[1] 

    #print(np.max(imGT), np.max(imBinary))

    TP = float(TP)/p
    TN = float(TN)/p
    FP = float(FP)/p
    FN = float(FN)/p
    SE = float(SE)/p

    confusionMatrix = [TP,FP,FN,TN,SE];

    return np.array(confusionMatrix)



def confusionMatrixToVar(confusionMatrix):
    TP = confusionMatrix[0];
    FP = confusionMatrix[1];
    FN = confusionMatrix[2];
    TN = confusionMatrix[3];
    SE = confusionMatrix[4];
    
    recall = TP / (TP + FN);
    specficity = TN / (TN + FP);
    FPR = FP / (FP + TN);
    FNR = FN / (TP + FN);
    PBC = 100.0 * (FN + FP) / (TP + FP + FN + TN);
    precision = TP / (TP + FP);
    FMeasure = 2.0 * (recall * precision) / (recall + precision);
    
    stats = np.array([recall,specficity,FPR,FNR,PBC,precision,FMeasure])

    return stats


def NumberOfDigits(Number):
    Count = 0
    while(Number > 0):
        Number = Number//10
        Count = Count + 1

    return Count

def saveList(myList,filename):
    # the filename should mention the extension 'npy'
    np.save(filename,myList)
    print("Saved successfully!")


def Calcul_metrics(datasets, results_path):

    for dataset in datasets:
        GT_path = '/home/simon/Desktop/Scene_Analysis/CDnet/dataset2014/dataset/nightVideos/' + dataset +'/groundtruth/img/'
        nb_images = len([name for name in os.listdir(GT_path)])
        print('number of images: ', nb_images)
        mypath = results_path + dataset 
        for f in os.listdir(mypath):
            #print(f)
            imgs_path = mypath + '/' + f + '/masks/img/'

            nb = '000001'
            results_ = []
            confusionMatrix = np.array([0.0,0.0,0.0,0.0,0.0])
            for i in range(nb_images):
                print(i)
                img_binary = cv2.imread(imgs_path + str(i) + '.jpg')[:,:,0]
                img_gt = cv2.imread(GT_path + 'gt'+str(nb) + '.png')[:,:,0]
                img_gt = cv2.resize(img_gt, (256,256))
                # print('Shapes: ', img_binary.shape)
                #print('Shapes: ', img_gt.shape)

                nb = int(nb)+1
                nb_digits = NumberOfDigits(int(nb))
                nb = str('0'*(6-nb_digits)) + str(nb)
                #print(nb)

                confusionMatrix += confusionMatrix_compare(img_binary, img_gt)

            confusionMatrix = confusionMatrix/nb_images
            stats = confusionMatrixToVar(confusionMatrix)
            results_.append(stats)
            saveList(results_, results_path + 'results/' + f + '.npy')
            print('F1-measure: ', stats[-1])
            print('recall, precision: ', stats[0], stats[-2])
            #print(confusionMatrix.shape)
            #break
            #assert(0)


def loadList(filenames):
    for filename in os.listdir(filenames):
        # the filename should mention the extension 'npy'
        tempNumpyArray=np.load(filenames + filename)[0,-1]
        if True:#filename.split('_')[-1:][0] == 'abandonedBox.npy':
            print(filename.split('_')[0:2], filename.split('_')[-2:], tempNumpyArray)
    #return tempNumpyArray.tolist()


results_path = 'output_CDnet2014/nightVideos/'
datasets = ['streetCornerAtNight']


#Calcul_metrics(datasets, results_path)
loadList(results_path + 'results/')



