import numpy as np
import os
from Read_flowFile import read_flow, flow_to_image
import cv2


##########################################################

if __name__ == '__main__':

	filename_Flows= '../../../Varna_datasets/Flows/'
	filename_Flows_imgs = '../../../Varna_datasets/FlowsImgs/img/'
	filename_Masks = '../../../Varna_datasets/FlowsMasks/img/'

	filename_Flows_npy = '../../../Varna_datasets/Flows_npy/flows/'

	nb_imgs = len(os.listdir(filename_Flows))
	print('number of images is ', nb_imgs)

	for nb in range(1, nb_imgs+1):
		print(nb)

		#img = cv2.imread(filename_Masks + str(nb) + '.jpg')
		#if img.any() == None:
		#	break

		
		flow_map = read_flow(filename_Flows + str(nb) + '.flo')
		#print(flow_map.shape, flow_map.dtype)
		#print(flow_map[500:510,:2,0])
		np.save(filename_Flows_npy + str(nb) + '.npy', flow_map)

		"""
		img = flow_to_image(flow_map)
		mask = np.where(img < 200, img, 0)

		tmp = mask[:,:,0] + mask[:,:,1] + mask[:,:,2]
		binary_mask = np.where(tmp == 0, 0, 1).astype(np.uint8)
		#print(binary_mask.dtype)

		cv2.imwrite(filename_Flows_imgs + str(nb) + '.jpg', img)
		cv2.imwrite(filename_Masks + str(nb) + '.jpg', binary_mask)
		"""
