
import numpy as np
from pyugm.factor import DiscreteFactor
from pyugm.factor import DiscreteBelief

import pickle
import matplotlib.pyplot as plt
import matplotlib
import seaborn
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize

from pyugm.model import Model
from pyugm.infer_message import LoopyBeliefUpdateInference
from pyugm.infer_message import FloodingProtocol


def reporter(infe, orde):
    for var in var_values.keys():
        marginal = infe.get_marginals(var)[0].data[0]
        var_values[var].append(marginal)
    change = orde.last_iteration_delta
    changes.append(change)
    energy = infe.partition_approximation()
    print ('{:3} {:8.2f} {:5.2f} {:8.2f}'.format(orde.total_iterations, change, marginal, energy))
 



# Specify the parameters (should be the same shape as the potential table would have been)
factor_parameters = np.array([['theta_0', 'theta_1'], ['theta_2', 'theta_3']])  
variables_names_and_cardinalities = [(1, 2), (2, 2)]
# Construct the factor
factor = DiscreteFactor(variables_names_and_cardinalities, parameters=factor_parameters)
print (factor)

# The factor still has its default potential table (all ones)
#factor.data

# Create a belief based on the factor
belief = DiscreteBelief(factor)

# Potentials are filled with the exponent of the parameters.
belief.set_parameters({'theta_0': np.log(2), 'theta_1': np.log(0.2), 'theta_2': np.log(5), 'theta_3': np.log(1)})
#belief.data

belief.set_parameters({'theta_0': np.log(1), 'theta_1': np.log(2), 'theta_2': np.log(1), 'theta_3': np.log(1)})
#belief.data

factor_parameters = np.array([['theta_0', 'theta_1'], ['theta_0', 'theta_0']])  
factor = DiscreteFactor([(1, 2), (2, 2)], parameters=factor_parameters)
belief = DiscreteBelief(factor)
belief.set_parameters({'theta_0': np.log(3), 'theta_1': np.log(5)})
print (belief.data)



seaborn.set_style("dark")
# Load pre-discretized image. Each pixel value is an integer between 0 and 32.
image = io.imread('varna_20190125_153327_0_900_0000001700.jpg')
#image = rgb2gray(image)
width = image.shape[0]
height = image.shape[1]
image = image[:,:,1]
image = resize(image, (image.shape[0] // 4, image.shape[1] // 4))

plt.figure(figsize=(14, 3))
plt.subplot(1, 2, 1)
_ = plt.imshow(image, cmap=matplotlib.cm.Greys_r, interpolation='nearest')
_ = plt.title('Image')
seaborn.set_style("darkgrid")
plt.subplot(1, 2, 2)
_ = plt.hist(image.flatten(), bins=32, color=(0.5, 0.5, 0.5))
_ = plt.title('Pixel intensity histogram')

plt.show()


observation_template = np.array([['obs_low'] * 300,
                                 ['obs_high'] * 300])
observation_template[0, 0:30] = 'obs_high'
observation_template[1, 0:30] = 'obs_low'


I, J = image.shape
factors = []
evidence = {}

# Add observation factors
for i in range(I):
    print(i)
    for j in range(J):
        
        label_variable_name = 'label_{}_{}'.format(i, j)
        observation_variable_name = 'obs_{}_{}'.format(i,j)
        factors.append(DiscreteFactor([(label_variable_name, 2), (observation_variable_name, 300)], parameters=observation_template))
        evidence[observation_variable_name] = image[i, j]




model = Model(factors)

order = FloodingProtocol(model, max_iterations=30)
inference = LoopyBeliefUpdateInference(model, order)

parameters = {'obs_high': 0.1, 'obs_low': -1.0}

inference.calibrate(evidence, parameters)


labels = np.zeros(image.shape)
for i in range(I):
    print(i)
    for j in range(J):
        
        variable_name = 'label_{}_{}'.format(i, j)
        label_factor = inference.get_marginals(variable_name)[0]
        labels[i, j] = label_factor.normalized_data[0] 

seaborn.set_style("dark")
plt.figure(figsize=(14, 3))
plt.subplot(1, 3, 1)
_ = plt.imshow(image, cmap=matplotlib.cm.Greys_r, interpolation='nearest')
_ = plt.title('Original image')
plt.subplot(1, 3, 2)
_ = plt.imshow(labels, cmap=matplotlib.cm.Greys, interpolation='nearest')
_ = plt.title('Inferred labels (white=foreground, black=background)')

plt.figure('results')
plt.imshow(labels)

plt.show()
newlabels = resize(labels, (width, height))
io.imsave('newmasks2.jpg',newlabels)

#---------------------------------------------------------------------------
'''
label_template = np.array([['same', 'different'], 
                           ['different', 'same']])

evidence = {}
factors = []

# Add observation factors
for i in range(I):
    print(i)
    for j in range(J):
        label_variable_name = 'label_{}_{}'.format(i, j)
        observation_variable_name = 'obs_{}_{}'.format(i, j)
        factors.append(DiscreteFactor([(label_variable_name, 2), (observation_variable_name, 32)], parameters=observation_template))
        evidence[observation_variable_name] = image[i, j] 
        
# Add label factors
for i in range(I):
    print(i)
    for j in range(J):
        variable_name = 'label_{}_{}'.format(i, j)
        if i + 1 < I:
            neighbour_down_name = 'label_{}_{}'.format(i + 1, j)
            factors.append(DiscreteFactor([(variable_name, 2), (neighbour_down_name, 2)], parameters=label_template))

print('model')
model = Model(factors)
print('model done')


# Get some feedback on how inference is converging by listening in on some of the label beliefs.
var_values = {'label_1_1': [], 'label_10_10': [], 'label_20_20': [], 'label_30_30': [], 'label_40_40': []}
changes = []
   
print('order')
order = FloodingProtocol(model, max_iterations=15)
print('order done')
inference = LoopyBeliefUpdateInference(model, order, callback=reporter)
print('infrence done')

parameters = {'same': 2.0, 'different': -1.0, 'obs_high': 1.0, 'obs_low': -0.0}


inference.calibrate(evidence, parameters)
print('calibrate done')

labels = np.zeros(image.shape)
for i in range(I):
    print(i)
    for j in range(J):
        variable_name = 'label_{}_{}'.format(i, j)
        label_factor = inference.get_marginals(variable_name)[0]
        labels[i, j] = label_factor.normalized_data[0] 



plt.figure(figsize=(14, 3))
plt.subplot(1, 3, 1)
_ = plt.imshow(image, cmap=matplotlib.cm.Greys_r, interpolation='nearest')
_ = plt.title('Original image')
plt.subplot(1, 3, 2)
_ = plt.imshow(labels, cmap=matplotlib.cm.Greys, interpolation='nearest')
_ = plt.title('Label beliefs \n(darker=higher background belief,\n lighter=higher foreground belief')
plt.subplot(1, 3, 3)
_ = plt.imshow(labels > 0.5, cmap=matplotlib.cm.Greys, interpolation='nearest')
_ = plt.title('Thresholded beliefs')

newlabels = resize(labels, (width, height))
io.imshow('newlabels',newlabels)
io.imsave('newmasks1.jpg',newlabels)

plt.show()

'''























































