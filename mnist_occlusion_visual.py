__author__ = 'kazem_safari'
# So here our strategy is basically the same.
# First, we load the network model.
# Second, we initialize all the tf.global_variables in a completely separate session.

# Lastly, here, instead of updating them with an optimization function and
# a training loop, we update them by restoring everything from the previously saved model

import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# This module itself depends on tensorflow:
from mnist_helper_functions import param_counter, mnist_inference

tf.reset_default_graph()
# placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

# reshape the input from 1D-vector to 2D-image
x_image = tf.reshape(x, [-1, 28, 28, 1])

# network params and hyperparams
nh_pre = 1  # number of channels input image
nh1 = 32  # number of units first convolutional layer
nh2 = 64  # number of units second convolutional layer
nh3 = 1024  # numebr of units first fully-connected layer
num_classes = 10  # number of output classes

# building network architecture function
[y_conv, cross_entropy, correct_prediction, accuracy, merged] = \
    mnist_inference(x_image, y_, keep_prob, nh1, nh2, nh3, num_classes)
#############################################################
###$$$$ Here the first step is completed!
#############################################################
# here since we are restoring a pre-trained model we DONOT need the optimizer function
# with tf.name_scope('adam_optimizer'):
#    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

##########################################################
###########################################################
# This intermediate step is crucial:

# i) instantiating an inti tensor
# Add the variable initializer Op.
init = tf.global_variables_initializer()

# ii) redefining a new instance of a sever object in tensorflow
# Create a saver.
saver = tf.train.Saver()

###########################################################
###########################################################

# path where the results of training are saved
model_path = 'mnist_trad/mnist_trad.ckpt'
save_path = model_path
result_dir = 'manyang'
sess = tf.InteractiveSession()


# with tf.Session() as sess:
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

# One key thing to remember, based on what i
# have read and what a ta had told me, is that,
# to restore a model completely,
# so that no part of our model would be missing,
# we need to access all these files simultaneously:
#####################
# mnist_trad.ckpt.meta
# mnist_trad.ckpt.index
# mnist_trad.ckpt.data-00000-of-00001
# checkpoint

# And the only way which is possible to do that is
# through saver.restore(sess, model_path)
# which accesses all these file altogether!
########################
####################################################
####################################################
# mnist_trad_restore()
from tensorflow.examples.tutorials.mnist import input_data
mnist0 = input_data.read_data_sets("MNIST_data/", one_hot=False)
mnist1 = input_data.read_data_sets("MNIST_data/", one_hot=True)
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline

from stack_occlusions import fixOcclusions_stackImages, fixImages_stackOcclusions



digitClass_indicies =[]
def list_digitClass_indicies(labels0_correctly_classified):
    #here we extract the indicies of each class from labels_correctly_classified
    for i in range(10):
        digitClass = np.nonzero(labels0_correctly_classified==i)
        #print(digitclass)
        digitClass_indicies.append(digitClass)
    #print(list_digitclass_indicies)
    return digitClass_indicies

#create a list of lists to store accuracies of stack of occluded images for each digit
occu_stack_acc = [[] for _ in range(10)]
#print(occu_stack_acc)
def occlusion_visual_analysis(patch_size, strides, imgs_correctly_classified,
                                labels1_correctly_classified, digitClass_indicies):
    iter_columns = range(0, 28-patch_size[0]+strides[0], strides[0]) #iterator along columns object
    iter_rows    = range(0, 28-patch_size[1]+strides[1], strides[1]) #iterator along rows object
    M1           = len(iter_columns)
    M2           = len(iter_rows)
    M            = M1*M2

    for d in range(10):
        #d =0
        digitClass         = digitClass_indicies[d][0] #get the indices of the digitclass == d
        print(type(digitClass))
        print(digitClass)
        L                  = digitClass.size
        print(L)
        rnd_digit          = np.random.randint(L, size=1)
        rnd_digit          = digitClass[int(rnd_digit)]
        print(rnd_digit)
        #print(imgs_correctly_classified[digitclass].shape)
        #images_placeholder = imgs_correctly_classified[digitclass]
        images             = imgs_correctly_classified[rnd_digit].reshape((-1,28,28))
        print(images.shape)

        true_label        = labels1_correctly_classified[rnd_digit].reshape((-1,10))
        print(true_label.shape)
        #perform occlusions
        givenOcclusions_stackImages = fixImages_stackOcclusions(images, strides, patch_size)[0]
        print(type(givenOcclusions_stackImages))
        print(givenOcclusions_stackImages.shape)
        givenOcclusions_stackImages = np.concatenate((images, givenOcclusions_stackImages), axis = 0)
        #M = len(givenOcclusions_stackImages)
        M = givenOcclusions_stackImages.shape[0]
        true_labels = np.tile(true_label, (M,1))
        print(true_labels.shape)
        summary_writer = tf.summary.FileWriter(result_dir + str(d))
        #compute accuracy
        occlusion_stack_placeholder = givenOcclusions_stackImages.reshape((-1,784))
        print(occlusion_stack_placeholder.shape)
        for i in range(occlusion_stack_placeholder.shape[0]):
            merged_str = sess.run(merged, feed_dict={x:occlusion_stack_placeholder[i].reshape((-1,784)),  y_: true_labels, keep_prob: 1.0})
            summary_writer.add_summary(merged_str,i)




N = 100
images       = mnist1.test.images[0:N]
#print(type(images))
print(images.shape)
labels0      = mnist0.test.labels[0:N] #one_hot = False
labels1      = mnist1.test.labels[0:N] #one_hot = True
print(labels1.shape)
#########################################################
### WARNING
#NEVER EVER NANME THE OUTPUT OF AN AVALUATION OF A TENSOR YOUR GRAPH TO THE NAME OF AN EXISTING
#TENSOR IN YOUR GRAPH. IT WILL CHANGE THE VALUE OF THAT TENSOR AND CAUSE HUGE PROBLEMS.
#SO, HERE WE CHOSE A DIFFERENT NAME FOR THE OUTPUT OF EVALUATIONS OF THE EXISTING TENSORS CORRECT_PREDICTION
# AND ACCURACY:
#######################################################
correct_pred, acc = sess.run([correct_prediction, accuracy],
                             feed_dict={x: images, y_: labels1, keep_prob: 1.0})
#np.set_printoptions(threshold=np.nan)
#print(correct_pred)
#print(acc)

#here we extract the correctly classified labels for occlusion analysis:
imgs_correctly_classified     = images[correct_pred.nonzero()]
labels0_correctly_classified  = labels0[correct_pred.nonzero()]
labels1_correctly_classified  = labels1[correct_pred.nonzero()]
#print(labels_correctly_classified)

digitClass_indicies = list_digitClass_indicies(labels0_correctly_classified)
strides      = [1, 1]
patch_size   = [5, 5]
occlusion_visual_analysis(patch_size, strides, imgs_correctly_classified,
                                labels1_correctly_classified, digitClass_indicies)




