#!/usr/bin/env python

##
# Massimiliano Patacchiola, Plymouth University 2016
#
# In this example I show you how to use a pretrained Deep Neural Network (DNN)
# for head pose estimation. It requires a tensorflow file containing the weights
# of the network, which are loaded at the beginning of the session.
#
# Attention: this example works with greyscale images of dimension 64x64 pixels

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
import cv2

# Create model
def multilayer_model(_X, _input0, _biases_input0, _hidden1, _biases_hidden1, _hidden2, _biases_hidden2, _output3, _biases_output3):
    
    _input0_result = tf.matmul(_X, _input0) + _biases_input0
    _hidden1_result = tf.nn.tanh(tf.matmul(_input0_result, _hidden1) + _biases_hidden1)
    _hidden2_result = tf.nn.tanh(tf.matmul(_hidden1_result, _hidden2) + _biases_hidden2)
    _output3_result = tf.nn.tanh(tf.matmul(_hidden2_result, _output3) + _biases_output3)
    return _output3_result

graph = tf.Graph()
with graph.as_default():
 
    print("Starting Graph creation...")
    
    # Variables
    image_size = 64
    num_hidden_units_1 = 256 
    num_hidden_units_2 = 256 
    num_hidden_units_3 = 256 
    num_labels = 3
    
    #0- the input placeholder
    tf_input = tf.placeholder(tf.float32,shape=(batch_size, image_size * image_size))
    
    #1- weights
    #tf.truncated_normal(shape, mean=0.0, stddev=1.0)
    weights_input0 = tf.Variable(tf.truncated_normal([image_size * image_size, num_hidden_units_1], 0.0, 1.0))    
    weights_hidden1 = tf.Variable(tf.truncated_normal([num_hidden_units_1, num_hidden_units_2], 0.0, 1.0))
    weights_hidden2 = tf.Variable(tf.truncated_normal([num_hidden_units_2, num_hidden_units_3], 0.0, 1.0)) 
    weights_output3 = tf.Variable(tf.truncated_normal([num_hidden_units_3, num_labels], 0.0, 1.0))    
    
    #2- biases
    biases_input0 = tf.Variable(tf.zeros([num_hidden_units_1]))
    biases_hidden1 = tf.Variable(tf.zeros([num_hidden_units_2]))    
    biases_hidden2 = tf.Variable(tf.zeros([num_hidden_units_3]))
    biases_output3 = tf.Variable(tf.zeros([num_labels]))
 
    #3- testing
    prediction = multilayer_model(tf_train_dataset, 
                     weights_input0, biases_input0, 
                     weights_hidden1, biases_hidden1, 
                     weights_hidden2, biases_hidden2, 
                     weights_output3, biases_output3)
    
    print("Finished.")

#Print the variables 
print("========== ALL TF VARS ======== ")
all_vars = tf.all_variables()
for k in all_vars:
      print(k.name)

#Load the checkpoint
ckpt = tf.train.get_checkpoint_state("./dnn_1600i_4h_3o")

#Create the session
_sess = tf.Session()
      
#Associate the weights stored in the checkpoint file to the
#local tensorflow variables      
tf.train.Saver(({"dnn_weights_input0": weights_input0, "dnn_biases_input0": biases_input0,
                 "dnn_weights_hidden1": weights_hidden1, "dnn_biases_hidden1": biases_hidden1,
                 "dnn_weights_hidden2": weights_hidden2, "dnn_biases_hidden2": biases_hidden2,
                 "dnn_weights_output3": weights_output3, "dnn_biases_output3": biases_output3
                 })).restore(_sess, ckpt.model_checkpoint_path)

#Load the image in greyscale with OpenCV
image = cv2.imread("image.jpg", 0) 
h,w = image.shape

#Resize the image if needed and get the predictions from the model
if(h == w and h>64):
    image_resized = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)
    image_normalised = np.add(image_resized, -127) #normalisation of the input
    feed_dict = {tf_input : image_normalised}
    predictions = _sess.run([prediction], feed_dict=feed_dict)
elif(h == w and h==64):
    image_normalised = np.add(image_resized, -127) #normalisation of the input
    feed_dict = {tf_input : image_normalised}
    predictions = _sess.run([prediction], feed_dict=feed_dict)
    print(predictions)
    #Here to see the output in degrees you should
    #multiply the first value inside prediction (roll) times 25
    #the second value in prediction (pitch) times 45
    #and the third value (yaw) times 90
else:
    raise ValueError('DnnHeadPoseEstimation: the image given as input is not squared or it is smaller than 64px.')

