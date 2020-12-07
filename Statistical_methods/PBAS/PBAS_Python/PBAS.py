import cv2
import numpy as np
import random
from math import ceil
import time
from queue import Queue
from threading import Thread


def initial_background(I_gray, N):
    
	I_pad = np.pad(I_gray, 1, 'symmetric')
	width = I_pad.shape[0]
	height = I_pad.shape[1]
	samples = np.zeros((N, 3, width, height))

	x = np.arange(width)
	y = np.arange(height)

	for n in range(N):
		#print(n)

		random_x = np.random.uniform(-1, +2, width)
		random_y = np.random.uniform(-1, +2, height)

		ri = random_x + x
		rj = random_y + y

		ri = ri.astype(int)
		rj = rj.astype(int)

		ri = np.where( (ri >= width) == True, width - 1, ri)
		ri = np.where( (ri < 0) == True, 0, ri)

		rj = np.where( (rj >= height) == True, height - 1, rj)
		rj = np.where( (rj < 0) == True, 0, rj)

		samples[n][0][ri] = I_pad[ri]
		samples[n][1][ri] = I_pad[ri]
		samples[n][2][ri] = I_pad[ri]

	samples = samples.T[1:height-1, 1:width-1]
	return samples.T


def Split_Filter(image):
	image = cv2.medianBlur(image,5)
	blur = cv2.GaussianBlur(image,(3,3),3)

	b = blur[:,:,0]
	g = blur[:,:,1]
	r = blur[:,:,2]

	gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

	return gray, b, g, r


def getFeatures(blurImage):
	ddepth = cv2.CV_32F 

	grad_x = cv2.Sobel(blurImage, ddepth, 1, 0, ksize=3, scale=1, delta=0)
	grad_y = cv2.Sobel(blurImage, ddepth, 0, 1, ksize=3, scale=1, delta=0)

	# magnitude and angle are 2 matrices that have the same dimensions of the input image
	magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees =True)

	descriptor = np.zeros((3, blurImage.shape[0], blurImage.shape[1]))
	descriptor[0] = magnitude
	descriptor[1] = angle
	descriptor[2] = blurImage

	return descriptor




def process(image, N, R, T, Q, D, Average_I_mag, backGroundFeatures, runs, queue_out):


	print(runs)

	################################################################
	# initialize background-model depending parameters
	# R
	R_scale = 5.0
	R_inc_dec = 0.05
	R_lower = 18

	# T	
	T_dec = 0.05
	T_inc = 1.0
	T_lower = 2
	T_upper = 200
	T_initial = 18


	nb_min = 1

	foreground = 255
	background = 0

	
	beta = 1.0
	alpha = 1.0
	

	width, height = image.shape[0], image.shape[1]

	# if(height < 10 or width < 10):
	#	print("Error: Occurrence of different image size in PBAS. STOPPING ")

	# elif(height == 0 or width == 0):
	#	print("Error: Width/height not set. Initialize first. width: "+ str(width) + " height: " + str(height) )

	##################################################################


	segMap = np.zeros((width, height))

	tempDistB = np.zeros((width, height))

	img_Features = getFeatures(image)


	# For the first N frames
	if(runs < N):
		
		backGroundFeatures[runs] = img_Features

		D[runs] = tempDistB

		if(runs == 0):
			R = np.ones((width, height)) * R_lower
			T = np.ones((width, height)) * T_initial
			Q = np.zeros((width, height))

		runs += 1

	# After N frames
	else:

		for ee in range(N-1):
			backGroundFeatures[ee] = backGroundFeatures[ee+1]

		backGroundFeatures[runs-1] = img_Features


	

	count = np.zeros((width, height))
	
	minDist = np.ones((width, height)) * 10000.0
	max_Norm = np.zeros((width, height))
	mean_minDist = np.zeros((width, height))

	glCounterFore = np.zeros((width, height))

	for k in range(0,runs-1):

		Norm = abs(backGroundFeatures[k][0] - img_Features[0])
		pixelValue = abs(backGroundFeatures[k][2] - img_Features[2])

		dist = (alpha/Average_I_mag) * Norm + beta * pixelValue

		temp_count = dist < R
		temp_count = temp_count.astype(int)
		
		count = np.add(count, temp_count)
		
		check = minDist > dist

		#minDist = np.where(check == True, dist, minDist)
		minDist[check == True] = dist[check == True]

		# used for average of Frame_mag
		max_Norm = np.where(dist > R, Norm, max_Norm)

		glCounterFore = np.where(dist > R, glCounterFore + 1.0, glCounterFore)

	temp_count = None
	check = None
	

	#################################################################
	##  Update Distance Matrix ##########################

	if (runs < N):
		D[runs] = np.where(count >= nb_min, minDist, D[runs])
		Q = np.add(Q, D[runs - 1])

	elif (runs == N):

		for ee in range(N-1):
			D[ee] = D[ee+1]
		
		D[runs-1] = np.where(count >= nb_min, minDist, 0.0)
		
		Q = np.add(Q, np.add(D[int(random.uniform(0,N))], (-1)*D[int(random.uniform(0,N))]))
				


	#######################################################

	mean_minDist = np.add(mean_minDist, np.sum(D, axis=0))


	# Update R    ########################################################	
	
	# is foreground
	condition1 = count < nb_min

	segMap[count < nb_min] = 255

	threshold_R1 = mean_minDist * R_scale

	threshold_R = R < threshold_R1
	threshold_R = threshold_R * condition1

	R = np.where(threshold_R == True, R + R * (R_inc_dec/10.0) , R)	
	
	threshold_R = R >= threshold_R1
	threshold_R = threshold_R * condition1

	R = np.where(threshold_R == True, R - R * (R_inc_dec/10.0) , R)	

	R = np.where(R < R_lower, R_lower, R)

	# is Background
	condition2 = (count >= nb_min)

	# mean_minDist[temp_count == True] = Q[temp_count == True]/float(runs)
	mean_minDist = np.where(condition2 == True, Q/float(runs), mean_minDist)
	
	threshold_R1 = mean_minDist * R_scale

	threshold_R = R < threshold_R1
	threshold_R = threshold_R * condition2

	# R[threshold_R == True] += R[threshold_R == True] * (R_inc_dec)
	R = np.where(threshold_R == True, R + R * (R_inc_dec) , R)	

	threshold_R = R >= threshold_R1
	threshold_R = threshold_R * condition2

	# R[threshold_R == True] -= R[threshold_R == True] * (R_inc_dec)
	R = np.where(threshold_R == True, R - R * (R_inc_dec) , R)	

	# R[R < R_lower] = R_lower
	R = np.where(R < R_lower, R_lower, R)


	mean_minDist = Q/float(runs)

	threshold_R1 = None
	threshold_R = None
	
	#########################################################################
	

	D = np.where(count >= nb_min, mean_minDist, D)


	# update pixel      ################################################
	prob = np.random.uniform(0,T, (width, height))
	condition3 = prob < 1

	temp_x, temp_y = np.where( (condition3 * condition2) == True)
	#print(temp_x.shape, temp_y.shape)
	
	#index_random = np.random.uniform(0,runs, len(temp_x) )
	
	temp_x = temp_x.astype(int)
	temp_y = temp_y.astype(int)
	#index_random = index_random.astype(int)

	if( len(temp_x) > 0):

		backGroundFeatures[runs - 1][0][temp_x][temp_y] = img_Features[0][temp_x][temp_y]
		# backGroundFeatures[runs - 1][1][temp_x][temp_y] = img_Features[1][temp_x][temp_y]
		backGroundFeatures[runs - 1][2][temp_x][temp_y] = img_Features[2][temp_x][temp_y]		


		# backGroundFeatures[index_random[0]][0][temp_x][temp_y] = img_Features[0][temp_x][temp_y]
		# backGroundFeatures[index_random[qq]][1][temp_x][temp_y] = img_Features[1][temp_x][temp_y]
		# backGroundFeatures[index_random[0]][2][temp_x][temp_y] = img_Features[2][temp_x][temp_y]
	

	temp_index = None
	prob = None
	###############################################################################



	# update Neighboor    ############################################
	prob = np.random.uniform(0,T, (width, height))
	condition4 = prob < 1

	temp_x, temp_y = np.where(condition2 * condition4 == True)

	xNeigh = temp_x + (np.random.uniform(-1, 2, temp_x.shape[0])).astype(int)
	yNeigh = temp_y + (np.random.uniform(-1, 2, temp_y.shape[0])).astype(int)

	xNeigh[xNeigh < 0] = 0
	xNeigh[xNeigh >= width] = width - 1

	yNeigh[yNeigh < 0] = 0
	yNeigh[yNeigh >= height] = height - 1
	

	#index = (np.random.uniform(0,runs, len(xNeigh) )).astype(int)

	#index = index.astype(int)
	xNeigh = xNeigh.astype(int)
	yNeigh = yNeigh.astype(int)

	if( len(xNeigh) > 0):
		
		backGroundFeatures[runs - 1][0][xNeigh][yNeigh] = img_Features[0][temp_x][temp_y]
		# backGroundFeatures[runs - 1][1][xNeigh][yNeigh] = img_Features[1][xNeigh][yNeigh]
		backGroundFeatures[runs - 1][2][xNeigh][yNeigh] = img_Features[2][temp_x][temp_y]


		# two methods
		# for qq in range(len(index)):

		# First: assign neighbors by pixels 
		# backGroundFeatures[index[0]][0][xNeigh][yNeigh] = img_Features[0][temp_x][temp_y]
		# backGroundFeatures[index[0]][1][xNeigh][yNeigh] = img_Features[1][xNeigh][yNeigh]
		# backGroundFeatures[index[0]][2][xNeigh][yNeigh] = img_Features[2][temp_x][temp_y]

		# Second: assign neighbors by their pixels
		# backGroundFeatures[index[0]][0][xNeigh][yNeigh] = img_Features[0][xNeigh][yNeigh]
		# backGroundFeatures[index[0]][1][xNeigh][yNeigh] = img_Features[1][xNeigh][yNeigh]
		# backGroundFeatures[index[0]][2][xNeigh][yNeigh] = img_Features[2][xNeigh][yNeigh]


	index = None
	temp_index = None
	prob = None
	temp_x = None
	temp_y = None
	xNeigh = None
	yNeigh = None
	####################################################################################3


	# Update T     #################################################33

	
	temp_T = T
	temp_T = np.where( segMap == 0, T - T_dec/(mean_minDist + 1), temp_T)
	temp_T = np.where( segMap == 255, T + T_inc/(mean_minDist + 1), temp_T)

	used_1 = temp_T < T_lower
	used_2 = temp_T > T_upper
	used_res = used_1 * used_2

	T = np.where(used_res == False, temp_T, T) 



	#####################################################################################

	# Update Average of Frame_mag    ####################################

	Average_I_mag = max_Norm/(glCounterFore+1.0)
	Average_I_mag = np.where(Average_I_mag > 20, Average_I_mag, 20)


	###################################################################################
	

	queue_out.put((True, runs, R, T, Q, D,segMap,Average_I_mag,backGroundFeatures))
	return True, runs, R, T, Q, D,segMap,Average_I_mag,backGroundFeatures

	

# Algorithm steps:			
# 1- read Video
# 2- gaussian Filter
# 3- split eaxh frame into 3 channels
# 4- Initilization of our model with zeros or using a simple algorithm
# 5- excute algorithm for each channel in parallel
# 6- bitwise_or between 3 outputs or bitwise_and between 3 outputs
# 7- repeat until end of video

N = 20
start = 0
nb_frames = 0

cap = cv2.VideoCapture('driveway-320x240.avi')



while(True):
#while(cap.isOpened()):
		
	start_time = time.time()
	
	# Capture frame-by-frame
	ret, frame = cap.read()

	# resize images
	dim = (320, 240)
	
	if(ret == True):
		frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA) 

		gray, b, g, r = Split_Filter(frame)
		
		width, height = gray.shape[0], gray.shape[1]
		
		# Initialization
		if (start == 0):

			# For gray 
			backGroundFeatures = np.zeros((N, 3, width, height))
			# backGroundFeatures = initial_background(gray, N)

			R = np.zeros((width, height))
			T = np.zeros((width, height))
	
			Q = np.zeros((width, height))
			D = np.zeros((N, width, height))
	
			Average_I_mag = np.ones((width, height))
			runs = 0


			# For red channel
			backGroundFeatures_r = np.zeros((N, 3, width, height))
			# backGroundFeatures_r = initial_background(r, N)

			R_r = np.zeros((width, height))
			T_r = np.zeros((width, height))
	
			Q_r = np.zeros((width, height))
			D_r = np.zeros((N, width, height))
	
			Average_I_mag_r = np.ones((width, height))
			runs_r = 0


			# For green channel
			backGroundFeatures_g = np.zeros((N, 3, width, height))
			# backGroundFeatures_g = initial_background(g, N)

			R_g = np.zeros((width, height))
			T_g = np.zeros((width, height))
	
			Q_g = np.zeros((width, height))
			D_g = np.zeros((N, width, height))
	
			Average_I_mag_g = np.ones((width, height))
			runs_g = 0


			# For bleu channel
			backGroundFeatures_b = np.zeros((N, 3, width, height))
			# backGroundFeatures_b = initial_background(b, N)

			R_b = np.zeros((width, height))
			T_b = np.zeros((width, height))
	
			Q_b = np.zeros((width, height))
			D_b = np.zeros((N, width, height))
	
			Average_I_mag_b = np.ones((width, height))
			runs_b = 0


			# First time only this part is executed
			start = 1
	
			## For low pass filter initialization
			# backGround_lowPassFilter = gray
		

		#ret, runs, R, T, Q, D, segMap, Average_I_mag, backGroundFeatures = process(gray, N, R, T, Q, D, Average_I_mag, backGroundFeatures, runs)
		
		#ret_r, runs_r, R_r, T_r, Q_r, D_r, segMap_r, Average_I_mag_r, backGroundFeatures_r = process(r, N, R_r, T_r, Q_r, D_r, Average_I_mag_r, backGroundFeatures_r, runs_r)

		#ret_g, runs_g, R_g, T_g, Q_g, D_g, segMap_g, Average_I_mag_g, backGroundFeatures_g = process(g, N, R_g, T_g, Q_g, D_g, Average_I_mag_g, backGroundFeatures_g, runs_g)

		#ret_b, runs_b, R_b, T_b, Q_b, D_b, segMap_b, Average_I_mag_b, backGroundFeatures_b = process(b, N, R_b, T_b, Q_b, D_b, Average_I_mag_b, backGroundFeatures_b, runs_b)
		
		
		queue_gray = Queue()
		thread_gray = Thread(target=process, args=[gray, N, R, T, Q, D, Average_I_mag, backGroundFeatures, runs, queue_gray])

		queue_r = Queue()
		thread_r = Thread(target=process, args=[r, N, R_r, T_r, Q_r, D_r, Average_I_mag_r, backGroundFeatures_r, runs_r, queue_r])

		queue_g = Queue()
		thread_g = Thread(target=process, args=[g, N, R_g, T_g, Q_g, D_g, Average_I_mag_g, backGroundFeatures_g, runs_g, queue_g])

		queue_b = Queue()
		thread_b = Thread(target=process, args=[b, N, R_b, T_b, Q_b, D_b, Average_I_mag_b, backGroundFeatures_b, runs_b, queue_b])



		thread_gray.start()
		thread_r.start()
		thread_g.start()
		thread_b.start()

		thread_gray.join()
		thread_r.join()
		thread_g.join()
		thread_b.join()

		ret, runs, R, T, Q, D, segMap, Average_I_mag, backGroundFeatures = queue_gray.get()

		ret_r, runs_r, R_r, T_r, Q_r, D_r, segMap_r, Average_I_mag_r, backGroundFeatures_r = queue_r.get()

		ret_g, runs_g, R_g, T_g, Q_g, D_g, segMap_g, Average_I_mag_g, backGroundFeatures_g = queue_g.get()

		ret_b, runs_b, R_b, T_b, Q_b, D_b, segMap_b, Average_I_mag_b, backGroundFeatures_b = queue_b.get()

		# result_and = cv2.bitwise_and( cv2.bitwise_and(segMap_r, segMap_g), segMap_b)
		result_or = cv2.bitwise_or( cv2.bitwise_and(segMap_r, segMap_g), segMap_b)
		

		##################################################################
		## Low Pass Filter applied on the output
		# alpha = 0.005
		# backGround_lowPassFilter = backGround_lowPassFilter * alpha + segMap * (1 - alpha)

		# cv2.imshow('low pass filter',backGround_lowPassFilter)
		##################################################################


		cv2.imshow('frame',frame)
		cv2.imshow('result_or',result_or)
		# cv2.imshow('result_and',result_and)
		cv2.imshow('result_gray',segMap)
		

	end_time = time.time()

	# Time elapsed
	seconds = end_time - start_time

	# Calculate frames per second
	fps  = nb_frames / seconds;
	print('fps:',fps)
	
	print('nb_frames:',nb_frames)
	nb_frames += 1

	print ('############################################################')

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


















