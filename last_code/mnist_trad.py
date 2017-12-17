__author__ = 'kazem_safari'

import os
import time
import tensorflow as tf
from mnist_helper_functions import mnist_inference, param_counter
# Load MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# In[2]:
# import matplotlib.pyplot as plt
# %matplotlib inline

# sample_image = mnist.train.next_batch(1)[0]
# print(sample_image.shape)

# sample_image = sample_image.reshape([28, 28])
# plt.imshow(sample_image, cmap='Greys')



def main():
    # Specify training parameters
    model_path = './mnist_trad/mnist_trad.ckpt'  # path where the training model is saved
    max_step = 10000  # the maximum iterations. After max_step iterations, the training will stop no matter what

    start_time = time.time()  # start timing

    ###############################
    # BUILDING THE NETWORK
    ################################
    tf.reset_default_graph()
    # placeholders for input data and input labeles
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)
    # reshape the input image
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    nh1 = 32
    nh2 = 64
    nh3 = 1024
    num_classes = 10

    # inference function
    (y_conv, cross_entropy, correct_prediction, accuracy, merged) = \
        mnist_inference(x_image, y_, keep_prob, nh1, nh2, nh3, num_classes)

    # define an optimizer function to setup training
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # start a session
    with tf.Session() as sess:
        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Run the Op to initialize the variables.
        sess.run(init)

        # start training
        for i in range(max_step + 1):
            batch = mnist.train.next_batch(50)  # make the data batch, which is used in the training iteration.
            # the batch size is 50
            if i % 200 == 0:
                # output the training accuracy every 200 iterations
                train_accuracy = sess.run(accuracy, feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))

            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})  # run one train_step

        # save the "mnist_trad" model after training is finished
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)

        # print test error
        print("test accuracy %g" % sess.run(accuracy, feed_dict={
            x: mnist.test.images[0:10000], y_: mnist.test.labels[0:10000], keep_prob: 1.0}))
        param_counter()

        # calculate and print the time it took to finish the training
        stop_time = time.time()
        print('The training takes %f second to finish' % (stop_time - start_time))


if __name__ == "__main__":
    main()
