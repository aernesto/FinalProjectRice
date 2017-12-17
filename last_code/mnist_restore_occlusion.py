
# coding: utf-8

# In[1]:


#Restoring the previously saved mnist model


# In[2]:


__author__  = 'kazem_safari'
# So here our strategy is basically the same.
# First, we load the network model. 
#Second, we initialize all the tf.global_variables in a completely separate session.

# Lastly, here, instead of updating them with an optimization function and
# a training loop, we update them by restoring everything from the previously saved model

import tensorflow as tf
import numpy as np
import os
import sys
import pickle


old_stdout = sys.stdout
log_file = open("results.log","w")
print("saving printed output into a log file: ")
sys.stdout = log_file

#This module itself depends on tensorflow:
from mnist_helper_functions import param_counter, mnist_inference

tf.reset_default_graph()
#placeholders
x           = tf.placeholder(tf.float32, shape = [None, 784])
y_          = tf.placeholder(tf.float32, shape = [None, 10])
keep_prob   = tf.placeholder(tf.float32)

#reshape the input from 1D-vector to 2D-image 
x_image     = tf.reshape(x, [-1, 28, 28, 1])

#network params and hyperparams
print('network params and hyperparams: ')
nh_pre      = 1    #number of channels input image
nh1         = 32   #number of units first convolutional layer
nh2         = 64   #number of units second convolutional layer
nh3         = 1024 #numebr of units first fully-connected layer
num_classes = 10   #number of output classes


print('number of channels input image: ')
print(nh_pre)
print('number of conv weight filters in first convolutional layer: ')
print(nh1)
print('number of conv weight filters in second convolutional layer: ')
print(nh2)
print('numebr of multiplicative weight filters in first fully-connected layer: ')
print(nh3)
print('numebr of multiplicative weight filters in last  fully-connected layer: ')
print(num_classes)

#building network architecture function
(y_conv, cross_entropy, accuracy, merged) = mnist_inference(x_image,y_, keep_prob, nh1, nh2, nh3, num_classes)
#############################################################
    ###$$$$ Here the first step is completed!
#############################################################
#here since we are restoring a pre-trained model we DONOT need the optimizer function
#with tf.name_scope('adam_optimizer'):
#    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

##########################################################
###########################################################
#This intermediate step is crucial:

    #i) instantiating an inti tensor
    # Add the variable initializer Op.
init  = tf.global_variables_initializer()        

    #ii) redefining a new instance of a sever object in tensorflow
    # Create a saver.
saver = tf.train.Saver()
###########################################################
###########################################################

# path where the results of training are saved
model_path = 'mnist_trad/mnist_trad.ckpt' 
save_path  = model_path

sess = tf.InteractiveSession()
#with tf.Session() as sess:
####################################################
####################################################
sess.run(init) 
###$$$$ Here the second step is completed!
####################################################
####################################################


####################################################
####################################################
saver.restore(sess, model_path)
print("Model restored from file: %s" % save_path)
###$$$$ Here the last step is completed!

#One key thing to remember, based on what i
#have read and what a ta had told me, is that, 
#to restore a model completely,
#so that no part of our model would be missing,
#we need to access all these files simultaneously:
#####################
#mnist_trad.ckpt.meta
#mnist_trad.ckpt.index
#mnist_trad.ckpt.data-00000-of-00001
#checkpoint

#And the only way which is possible to do that is
#through saver.restore(sess, model_path)
#which accesses all these file altogether!
########################
####################################################
####################################################
#mnist_trad_restore()
#########################################################
### Caviat
#NEVER EVER NANME THE OUTPUT OF AN AVALUATION OF A TENSOR YOUR GRAPH TO THE NAME OF AN EXISTING
#TENSOR IN YOUR GRAPH. IT WILL CHANGE THE VALUE OF THAT TENSOR AND CAUSE HUGE PROBLEMS.
#SO, HERE WE CHOSE A DIFFERENT NAME FOR THE OUTPUT OF EVALUATIONS OF THE EXISTING TENSORS CORRECT_PREDICTION
# AND ACCURACY:
#######################################################


# In[3]:


#loading the mnist dataset


# In[4]:


from tensorflow.examples.tutorials.mnist import input_data
mnist0 = input_data.read_data_sets("MNIST_data/", one_hot=False)
mnist1 = input_data.read_data_sets("MNIST_data/", one_hot=True)

N = 10000
print("number of images for testing: ")
print(N)

images       = mnist1.test.images[0:N]
images       = images.reshape((-1, 28, 28))
#print(type(images))
#print(images.dtype)
print("shape of test images array: ")
print(images.shape)
labels0      = mnist0.test.labels[0:N] #one_hot = False
labels1      = mnist1.test.labels[0:N] #one_hot = True
print("shape of labels array: ")
#print(labels1.dtype)
print(labels1.shape)
#print(labels0.dtype)
#print(labels0.shape)


# In[5]:


#mnist_patch_occlusion_analysis


# In[6]:


from mnist_helper_functions import get_inds
from stack_occlusions       import get_occ_dims, get_patch_pins, fixOcclusions_stackImages
from array_helper_functions import ind2sub, ind2sub_all, sub2ind, sub2ind_all, min_array2d, max_array2d 
########################################################################################
inds_dict = get_inds(labels0)
##############################
def get_prediction_np(images, labels1, keep_prob_float):
    net_output      = sess.run(y_conv, feed_dict = 
                              {x        : images.reshape((-1, 784)),
                               y_       : labels1,
                               keep_prob: keep_prob_float})
    #print("net_output: ")
    #print("shape: ", net_output.shape)
    #print(net_output)
    #print(2*'\n')
    
    net_pred_label  = np.argmax(net_output, axis = 1)
    #print("net_pred_label: ")
    #print("shape: ", net_pred_label.shape)
    #print(net_pred_label)
    #print(2*'\n')
    
    true_label      = np.argmax(labels1, axis = 1)
    #print("true_label: ")
    #print("shape: ", true_label.shape)
    #print(true_label)
    #print(2*'\n')
    
    correct_pred_bool = np.equal(net_pred_label, true_label)
    #print("correct_pred_bool: ")
    #print(correct_pred_bool.shape)
    #print(correct_pred_bool)
    #print(2*'\n')
    
    correct_pred = np.uint32(correct_pred_bool)
    #print('correct_pred')
    #print(correct_pred.shape)
    #print(correct_pred)
    
    accuracy     = np.mean(correct_pred)
    #print('accuracy')
    #print(accuracy)
    return (net_pred_label, accuracy)

def get_accuracy_tf(images, one_hot_labels, keep_prob_float):
    images_placeholder = images.reshape(-1,images.shape[1]*images.shape[2])
    acc                = sess.run(accuracy, 
                                  feed_dict={x         : images_placeholder,
                                             y_        : one_hot_labels, 
                                             keep_prob : keep_prob_float})
    return acc

def min_max_acc_val_pins(acc_arr, pins_arr):
    (min_ac_val, min_ac_ind, min_ac_sub)  = min_array2d(acc_arr)
    (max_ac_val, max_ac_ind, max_ac_sub)  = max_array2d(acc_arr)
    min_acc_arr = np.array([[min_ac_val, min_ac_ind]])
    max_acc_arr = np.array([[max_ac_val, max_ac_ind]])
    print("in occlusion accuracy matrix printed above:")
    print("min accuracy value: ", min_ac_val)
    print("min accuracy (i,j)-indices: ")
    print(min_ac_sub)
    print("max accuracy: ", max_ac_val)
    print("max accuracy (i,j)-indices: ")
    print(max_ac_sub)
    print('\n')
    min_pins_arr = pins_arr[min_ac_ind] 
    max_pins_arr = pins_arr[max_ac_ind]
    print("min accuracy patch pins (i,j)-pixel indices on image:")
    print(min_pins_arr)
    print("max accuracy patch pins (i,j)-pixel indices on image:")
    print(max_pins_arr)
    print('\n')
    return (min_acc_arr, np.uint32(min_pins_arr), max_acc_arr, np.uint32(max_pins_arr))
##################################################################################
def occlusion_accuracy_analyzer(images, labels1, patch_size, strides, sess):
    (M1,M2) = get_occ_dims(strides, patch_size)
    M       = M1*M2
    acc_arr = np.zeros((M1, M2))
    #perform occlusions
    givenOcclusions_stackImages = fixOcclusions_stackImages(images, strides, patch_size)
    #M = len(givenOcclusions_stackImages)
    sub_list   = ind2sub_all((M1,M2))
    #run "accuracy" node in the computational graph for each stack of "givenOcclusions_stackImages"
    for i in range(M): #we want to eval accuracy for each (occluded) stack seperately  
        (_, acc_arr[sub_list[i][0], sub_list[i][1]]) =         get_prediction_np(givenOcclusions_stackImages[i], labels1, 1.0)
  
    print(np.around(acc_arr, decimals = 2))
    print('\n')
    return np.around(acc_arr, decimals = 2)

def occlusion_accuracy_analyzer_digitwise(images, labels1, inds_dict, patch_size, strides):
    (M1, M2)    = get_occ_dims(strides, patch_size)
    
    #create an array to store accuracies of stack of occluded images per digit class 
    acc_arr     = np.zeros((10,M1,M2))
    #storing the value and index of min and maxaccuracy for each digit
    min_acc_arr = np.zeros((10,2))
    max_acc_arr = np.zeros((10,2))
    
    #pins array
    print(images.shape)
    arr_shape     = [images.shape[1], images.shape[2]]
    pins_arr      = get_patch_pins(arr_shape, strides, patch_size)
    #get pins, i.e. the (i,j)-index of 4 corners of each patch
    #the (i,j)-indicies of the 4 corners of the patch where min accuracy and max accuracy happens
    min_pins_arr  = np.zeros((10, 2, 2))  
    max_pins_arr  = np.zeros((10, 2, 2))
    for d in range(10):
        print('digit %d: '%d)
        inds        = inds_dict[d] #get the indices of all digit = d 
        images_d    = images[inds]
        labels_d    = labels1[inds]

        acc_arr[d]  = occlusion_accuracy_analyzer(images_d, labels_d, patch_size, strides, sess)
        
        #min and max accuracy values, indices and their patch pins in our images for acc_array
        #This gives us the most and least vulnarable locations in the image to occlusions
        #val and index
        (min_acc_arr[d], min_pins_arr[d],         max_acc_arr[d], max_pins_arr[d]) = min_max_acc_val_pins(acc_arr[d], pins_arr)
    
    return (acc_arr, min_acc_arr, np.uint32(min_pins_arr), max_acc_arr, np.uint32(max_pins_arr))
##########################################################################################
#one-to-one correspondence
#min_pin  <---------------------> digit 
## extract the labels that network classifies to, for each min_pin corresponding to each digit



# In[7]:


######################################################################
######################################################################
#Accuracy-Occlusion Analysis restricted to correctly_classified_digits
######################################################################
######################################################################


# In[8]:


######################################################################
######################################################################
#Accuracy-Occlusion Analysis on the entire test set, i.e. not restricting to correctly classified images
######################################################################
######################################################################


# In[9]:


########################################
########################################
#before occlusion
print("before occlusion:")
#accuracy on entire test set
print("accuracy on test set with no occlusion : ")
print(np.around(sess.run(accuracy, feed_dict={x:images.reshape((-1,784)),
                                    y_       : labels1, 
                                    keep_prob: 1.0}), decimals = 2))
print('\n')
#accuracy per class
acc_arr = np.zeros((10,1))
for d in range(10):
    inds_d     = inds_dict[d]
    acc_arr[d] = sess.run(accuracy, feed_dict={x        : images[inds_d].reshape((-1,784)),
                                               y_       : labels1[inds_d], 
                                               keep_prob: 1.0})
print("accuracy per class on test set with no occlusion : ")
print(np.around(acc_arr, decimals = 2))
np.save('acc_arr', np.around(acc_arr, decimals = 2))
print(2*'\n')

#################################
###################################
#after occlusion
print("after occlusion")
strides      = [1, 1]
patch_size   = [5, 5]
arr_shape    = [images.shape[1], images.shape[2]]
pins_arr     = get_patch_pins(arr_shape, strides, patch_size)

#accuracy on entire test set
print("accuracy per patch on whole test set : ")
acc_arr_entire = occlusion_accuracy_analyzer(images, labels1, patch_size, strides, sess)
(min_acc_arr_entire, min_pins_arr_entire, max_acc_arr_entire, max_pins_arr_entire) = min_max_acc_val_pins(acc_arr_entire, pins_arr)



#accuracy per class
print("accuracy per patch per class on test set : ")
#Analyzing the effect of occlusion by a moving patch on the entire dataset
(acc_arr, min_acc_arr, min_pins_arr, max_acc_arr, max_pins_arr) = occlusion_accuracy_analyzer_digitwise(images, labels1, inds_dict, patch_size, strides)
np.save('acc_arr.npy', acc_arr)
np.save('min_acc_arr.npy', min_acc_arr)
np.save('min_pins_arr.npy', min_pins_arr)
np.save('max_acc_arr.npy', max_acc_arr)
np.save('max_pins_arr.npy', max_pins_arr)
#############################################################################################
#############################################################################################


# In[19]:


def get_occ_preds_labels(images, labels1, inds_dict, min_pins_arr):
    #what are the predicted labels from the network by applying
    #occlusion at the specific min pins, which depends on the digit.
    
    #In other words, fix the min pins location for the occlusion patch.
    #Then get the predicted labels of the network for the corresponding digit. 
    
    pred_labels = {}
    acc = {}
    for d in range(10):
        inds         = inds_dict[d] 
        print("total number of %d's in test set: "%d)
        print(len(inds))
        
        occ_images   = images[inds].copy()
        true_labels  = labels1[inds].copy()
        
        pin          = min_pins_arr[d]
        print("min accuracy patch pins (i,j)-pixel indices on %d's:"%d)
        print(pin)
        
        
        #perform occlusion
        occ_images[:, pin[0,0]: pin[0,1], pin[1,0]: pin[1,1]] = 0

        #run "predicted_label" node the computational graph
        (pred_labels[d], acc[d])= get_prediction_np(occ_images, true_labels, 1.0)    
        print("acc for digit %d: "%d)
        print(acc[d])
        print('\n')
        print("pred_labels for digit %d: "%d)
        print(pred_labels[d])
    return pred_labels

pred_labels = get_occ_preds_labels(images, labels1, inds_dict, min_pins_arr)

output = open('pred_labels.pkl', 'wb')
pickle.dump(pred_labels, output)
output.close()
sys.stdout = old_stdout
log_file.close()

