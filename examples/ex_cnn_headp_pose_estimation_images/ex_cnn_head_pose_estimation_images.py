#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import tensorflow as tf
import cv2
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator

sess = tf.Session() #Launch the graph in a session.
my_head_pose_estimator = CnnHeadPoseEstimator(sess) #Head pose estimation object
my_head_pose_estimator.load_yaw_variables(os.path.realpath("../../etc/tensorflow/head_pose/yaw/cnn_cccdd_30k"))

for i in range(1,9):
    file_name = str(i) + ".jpg"
    print("Processing image ..... " + file_name)
    image = cv2.imread(file_name) #Read the image with OpenCV
    yaw = my_head_pose_estimator.return_yaw(image) #Evaluate the yaw angle using a CNN
    print("Estimated yaw ..... " + str(yaw[0,0,0]))
    print("")
