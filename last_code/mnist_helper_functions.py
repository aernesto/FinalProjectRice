
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
#helper_functions
#________________________________________________
def weight_variable(shape):
    initial_W = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial_W)

def bias_variable(shape):
    initial_b = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_b)

def conv2d(x, W):
    h_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return h_conv

def max_pool_2x2(x):
    h_max = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    return h_max

#counting the total number of parameters in our network
def param_counter():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        #print(shape)
        #print(len(shape))
        variable_parameters = 1
        for dim in shape:
            #print(dim)
            variable_parameters *= dim.value
        #print(variable_parameters)
        total_parameters += variable_parameters
    print("total number of parameters:", total_parameters)

def get_inds(labels0):
    #input: 
            #labels0: labels must be with one_hot = False
    
    #output: 
            #inds: a dictionary consisting of labels belonging to each class
    inds = {}
    for d in range(10):
        inds[d] = np.nonzero(labels0==d)
    return inds

def mnist_inference(x_image, y_, keep_prob, nh1, nh2, nh3, num_classes):
    # first convolutional layer
    #nh1 = 32
    input_channels = 1
    with tf.name_scope('conv1'):
        W_conv1 = tf.get_variable('W_conv1', shape = [5, 5, input_channels, nh1], 
                                    initializer = tf.contrib.layers.xavier_initializer(),
                                    dtype=tf.float32)
        b_conv1 = bias_variable([nh1])

        h_conv1 = conv2d(x_image, W_conv1) + b_conv1
        h_bnorm1= tf.contrib.layers.batch_norm(h_conv1, epsilon=1e-5, scope='bn1')
        h_act1  = tf.nn.relu(h_bnorm1)
        
        h_act1_perm = tf.transpose(h_act1,perm=[3,1,2,0])
        h_act1_imgs = tf.summary.image('h_act1_imgs',h_act1_perm)
        
        with tf.name_scope('conv1_output'):
            h_pool1 = max_pool_2x2(h_act1)

    # second convolutional layer
    #nh2 = 64
    with tf.name_scope('conv2'):
        W_conv2 = tf.get_variable('W_conv2', shape = [5, 5, nh1, nh2], 
                                    initializer = tf.contrib.layers.xavier_initializer(),
                                    dtype=tf.float32)
        b_conv2 = bias_variable([nh2])
        h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
        h_bnorm2= tf.contrib.layers.batch_norm(h_conv2, epsilon=1e-5, scope='bn2')
        h_act2  = tf.nn.relu(h_bnorm2)
        with tf.name_scope('conv2_output'):
            h_pool2 = max_pool_2x2(h_act2)

    # densely connected layer
    #nh3 = 1024
    with tf.name_scope('fc1'):
        W_fc1 = tf.get_variable('W_fc1', shape = [7 * 7 * nh2, nh3], 
                                 initializer = tf.contrib.layers.xavier_initializer(),
                                 dtype=tf.float32)
        b_fc1 = bias_variable([nh3])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * nh2])
        with tf.name_scope('fc1_output'):
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    with tf.name_scope('dropout'):
        with tf.name_scope('dropout_output'):
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # softmax
    #num_classes = 10
    with tf.name_scope('fc2'):
        W_fc2 = tf.get_variable('W_fc2', shape =[nh3, num_classes], 
                                 initializer = tf.contrib.layers.xavier_initializer(),
                                 dtype=tf.float32)
        b_fc2 = bias_variable([num_classes])
        with tf.name_scope('net_output'):
            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    #loss function
    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    
    #correct_predictions and accuracy
    with tf.name_scope('predictions'):
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    merged = tf.summary.merge_all()
    return (y_conv, cross_entropy, accuracy, merged)

