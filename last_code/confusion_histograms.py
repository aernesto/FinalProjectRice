import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from mnist_helper_functions import get_inds
import matplotlib.pyplot as plt
import pickle
import matplotlib


font = {'family' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

# mnist0 = input_data.read_data_sets("MNIST_data/", one_hot=False)
# N = 110
# images       = mnist0.test.images[0:N]
# images       = images.reshape((-1, 28, 28))
# labels0      = mnist0.test.labels[0:N] #one_hot = False

# inds_dict = get_inds(labels0)

pkl_file    = open('pred_labels.pkl', 'rb')
pred_labels = pickle.load(pkl_file)
pkl_file.close()

min_pins_arr = np.load('min_pins_arr.npy')

for d in range(10):
#         inds         = inds_dict[d]
#         print("total number of %d's in test set: "%d)
#         print(len(inds))
#         occ_image    = images[inds].copy()[0]
#         pin          = min_pins_arr[d]
#         print("min accuracy patch pins (i,j)-pixel indices on %d's:"%d)
#         print(pin)
    #perform occlusion
#         occ_image[pin[0,0]: pin[0,1], pin[1,0]: pin[1,1]] = 0
#         plt.imshow(occ_image)
#         plt.show()
    plt.figure()
    plt.hist(pred_labels[d], normed=False, bins=[-.5,.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5], align='mid', log=True)
    plt.title("outputs for digit %d"%d)
    plt.xlabel("predicted label")
    plt.xlim([-1,10])
    plt.xticks( np.arange(10) )
    plt.ylabel("Frequency")
    plt.show()
