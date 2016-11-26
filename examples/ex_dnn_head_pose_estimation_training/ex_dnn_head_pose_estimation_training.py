#!/usr/bin/env python

##
# Massimiliano Patacchiola, Plymouth University 2016
#
# In this example I show you how to train a Deep Neural Network (DNN)
# for the head pose estimation. It requires a pickle file containing
# some numpy matrix, representing the images and the labels necessary
# for the training (see ex_aflw_parser.py). I cannot add the pickle file 
# here because the AFLW dataset has a license which not allow to further  
# copy, publish or distribute any portion of the AFLW database.
# Go on the original website and ask for a free registration to have it:
# https://lrs.icg.tugraz.at/research/aflw/
#
# Attention: this example works with greyscale images of dimension 64x64 pixels
# The labels should be Roll, Pitch and Yaw normalised in the range [-1, +1]
# Roll is in range [-25, +25]
# Pitch is in range [-45, +45]
# Yaw is in range [-90, +90]


# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

#Load the standard file
pickle_file = 'aflw_dataset.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  training_dataset = save['training_dataset']
  training_label = save['training_label']
  validation_dataset = save['validation_dataset']
  validation_label = save['validation_label']
  test_dataset = save['test_dataset']
  test_label = save['test_label']
  del save  # hint to help gc free up memory
  print('Training set', training_dataset.shape, training_label.shape)
  print('Validation set', validation_dataset.shape, validation_label.shape)
  print('Test set', test_dataset.shape, test_label.shape)
  #Normalising the images
  training_dataset -= 127
  validation_dataset -= 127
  test_dataset -= 127

def final_test(predictions, labels):

    row, col = labels.shape

    roll_prediction = np.copy(predictions[:,0])
    pitch_prediction = np.copy(predictions[:,1])
    yaw_prediction = np.copy(predictions[:,2])
    roll_labels = np.copy(labels[:,0])
    pitch_labels = np.copy(labels[:,1])
    yaw_labels = np.copy(labels[:,2])

    #To degree
    roll_prediction *= 25 # to degree
    pitch_prediction *= 45 # to degree
    yaw_prediction *= 90 # to degree

    #Root mean square error
    roll_mean_error = np.sum(np.square(roll_prediction - roll_labels)) * 1/row
    roll_mean_error = np.sqrt(roll_mean_error)
    pitch_mean_error = np.sum(np.square(pitch_prediction - pitch_labels)) * 1/row
    pitch_mean_error = np.sqrt(pitch_mean_error)
    yaw_mean_error = np.sum(np.square(yaw_prediction - yaw_labels)) * 1/row
    yaw_mean_error = np.sqrt(yaw_mean_error)

    print("=== TEST MEAN ERROR (DEGREE) ===")
    print("ROLL: " + str(roll_mean_error))
    print("PITCH: " + str(pitch_mean_error))
    print("YAW: " + str(yaw_mean_error))



def accuracy(predictions, labels, verbose=False):
  #Roll
  N, col = labels.shape
  if(verbose == True):
      print("PRED:   " + str(predictions[0,:]))
      print("LABEL:  " + str(labels[0,:]))

      #First value
      pred_value = np.copy(predictions[0,2])
      real_value = np.copy(labels[0,2])
      pred_value *= 25 # to degree
      real_value *= 25 # to degree
      print("=========")
      print("PRED ROLL: " + str(pred_value))
      print("REAL ROLL: " + str(real_value))

  prediction_copy = np.copy(predictions[:,2])
  labels_copy = np.copy(labels[:,2])
  from sklearn.metrics import mean_squared_error
  RMSE = mean_squared_error(labels_copy, prediction_copy)**0.5
  return RMSE


# Create model
def multilayer_model(_X, _input0, _biases_input0, _hidden1, _biases_hidden1, _hidden2, _biases_hidden2, _output3, _biases_output3):   
    _input0_result = tf.matmul(_X, _input0) + _biases_input0
    _hidden1_result = tf.nn.tanh(tf.matmul(_input0_result, _hidden1) + _biases_hidden1)
    _hidden2_result = tf.nn.tanh(tf.matmul(_hidden1_result, _hidden2) + _biases_hidden2)
    _output3_result = tf.nn.tanh(tf.matmul(_hidden2_result, _output3) + _biases_output3)
    return _output3_result

batch_size = 128 #was 128
graph = tf.Graph()
with graph.as_default():
 
    print("Starting Graph creation...")
    
    # Variables
    image_size = 64
    num_hidden_units_1 = 256 
    num_hidden_units_2 = 256 
    num_hidden_units_3 = 256 
    num_labels = 3
    
    #0- datasets
    tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(validation_dataset)
    #Sanitized version
    tf_test_dataset = tf.constant(test_dataset)
    
    #1- weights
    #tf.truncated_normal(shape, mean=0.0, stddev=1.0)
    weights_input0 = tf.Variable(tf.truncated_normal([image_size * image_size, num_hidden_units_1], 0.0, 1.0))    
    weights_hidden1 = tf.Variable(tf.truncated_normal([num_hidden_units_1, num_hidden_units_2], 0.0, 1.0))
    weights_hidden2 = tf.Variable(tf.truncated_normal([num_hidden_units_2, num_hidden_units_3], 0.0, 1.0)
    weights_output3 = tf.Variable(tf.truncated_normal([num_hidden_units_3, num_labels], 0.0, 1.0))    
    
    #2- biases
    biases_input0 = tf.Variable(tf.zeros([num_hidden_units_1]))
    biases_hidden1 = tf.Variable(tf.zeros([num_hidden_units_2]))    
    biases_hidden2 = tf.Variable(tf.zeros([num_hidden_units_3]))
    biases_output3 = tf.Variable(tf.zeros([num_labels]))
 
    #3- Defining a variable for saving the session parameters
    saver = tf.train.Saver({'dnn_weights_input0': weights_input0, 
                            'dnn_weights_hidden1': weights_hidden1,   
                            'dnn_weights_hidden2': weights_hidden2,
                            'dnn_weights_output3': weights_output3,
                            'dnn_biases_input0': biases_input0,
                            'dnn_biases_hidden1': biases_hidden1,
                            'dnn_biases_hidden2': biases_hidden2,
                            'dnn_biases_output3': biases_output3})   
    #4- training
    train_prediction = multilayer_model(tf_train_dataset, 
                     weights_input0, biases_input0, 
                     weights_hidden1, biases_hidden1, 
                     weights_hidden2, biases_hidden2, 
                     weights_output3, biases_output3)
    
    # Minimize the squared errors.
    loss = tf.reduce_mean(tf.square(train_prediction - tf_train_labels))


    #5- Adding the regularization terms to the loss
    beta = 5e-4
    loss += (beta * tf.nn.l2_loss(weights_input0)) 
    loss += (beta * tf.nn.l2_loss(weights_hidden1)) 
    loss += (beta * tf.nn.l2_loss(weights_hidden2)) 
    loss += (beta * tf.nn.l2_loss(weights_output3))
    
    #6- Optimizer.
    learning_rate = 0.001
    global_step = tf.Variable(0)  # count the number of steps taken.
    #learning_rate = tf.train.exponential_decay(0.5, global_step, 5000, 0.96, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step) 
        
    #Valid prediction
    valid_prediction = multilayer_model(tf_valid_dataset, 
                     weights_input0, biases_input0, 
                     weights_hidden1, biases_hidden1, 
                     weights_hidden2, biases_hidden2, 
                     weights_output3, biases_output3)
    #Test prediction
    test_prediction = multilayer_model(tf_test_dataset, 
                     weights_input0, biases_input0, 
                     weights_hidden1, biases_hidden1, 
                     weights_hidden2, biases_hidden2, 
                     weights_output3, biases_output3)
    
    print("Finished.")



num_steps = 50001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Starting Training...")
  print("Train dataset shape: ", training_dataset.shape)
  for step in range(num_steps):
    # Pick an offset within the training data
    offset = (step * batch_size) % (training_label.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = training_dataset[offset:(offset + batch_size), :]
    batch_labels = training_label[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 1000 == 0):
        print("Minibatch loss at step %d: %f" % (step, l))
        print("Minibatch accuracy: " + str(accuracy(predictions, batch_labels)))
        print("Validation accuracy: " + str(accuracy(valid_prediction.eval(), validation_label, True)))      
  
  saver.save(session, './dnn_1600i_4h_3o', global_step=step) #save the session    
  print("Test accuracy: " + str(accuracy(test_prediction.eval(), test_label)))
  final_test(test_prediction.eval(), test_label)

