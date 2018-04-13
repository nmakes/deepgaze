#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#In this example the Deepgaze CNN head pose estimator is used to get the YAW angle.
#The angle is projected on the input images and showed on-screen as a red line.
#The images are then saved in the same folder of the script.

import numpy as np
import os
import tensorflow as tf
import cv2
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator

#Function used to get the rotation matrix
def yaw2rotmat(yaw):
    x = 0.0
    y = 0.0
    z = yaw
    ch = np.cos(z)
    sh = np.sin(z)
    ca = np.cos(y)
    sa = np.sin(y)
    cb = np.cos(x)
    sb = np.sin(x)
    rot = np.zeros((3,3), 'float32')
    rot[0][0] = ch * ca
    rot[0][1] = sh*sb - ch*sa*cb
    rot[0][2] = ch*sa*sb + sh*cb
    rot[1][0] = sa
    rot[1][1] = ca * cb
    rot[1][2] = -ca * sb
    rot[2][0] = -sh * ca
    rot[2][1] = sh*sa*cb + ch*sb
    rot[2][2] = -sh*sa*sb + ch*cb
    return rot


sess = tf.Session() #Launch the graph in a session.
my_head_pose_estimator = CnnHeadPoseEstimator(sess) #Head pose estimation object
# Load the weights from the configuration folders
my_head_pose_estimator.load_yaw_variables(os.path.realpath("../../etc/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf"))
my_head_pose_estimator.load_roll_variables(os.path.realpath("../../etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf"))
my_head_pose_estimator.load_pitch_variables(os.path.realpath("../../etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf"))

for i in range(1,9):
    file_name = str(i) + ".jpg"
    file_save = str(i) + "_axes.jpg"
    print("Processing image ..... " + file_name)
    #file_name = "1.jpg"
    image = cv2.imread(file_name)
    cam_w = image.shape[1]
    cam_h = image.shape[0]
    c_x = cam_w / 2
    c_y = cam_h / 2
    f_x = c_x / np.tan(60/2 * np.pi / 180)
    f_y = f_x
    camera_matrix = np.float32([[f_x, 0.0, c_x],
                                [0.0, f_y, c_y], 
                                [0.0, 0.0, 1.0] ])
    print("Estimated camera matrix: \n" + str(camera_matrix) + "\n")
    #Distortion coefficients
    camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])
    #Defining the axes
    axis = np.float32([[0.0, 0.0, 0.0], 
                       [0.0, 0.0, 0.0], 
                       [0.0, 0.0, 0.5]])

    roll_degree = my_head_pose_estimator.return_roll(image, radians=False)  # Evaluate the roll angle using a CNN
    pitch_degree = my_head_pose_estimator.return_pitch(image, radians=False)  # Evaluate the pitch angle using a CNN
    yaw_degree = my_head_pose_estimator.return_yaw(image, radians=False)  # Evaluate the yaw angle using a CNN
    print("Estimated [roll, pitch, yaw] (degrees) ..... [" + str(roll_degree[0,0,0]) + "," + str(pitch_degree[0,0,0]) + "," + str(yaw_degree[0,0,0])  + "]")
    roll = my_head_pose_estimator.return_roll(image, radians=True)  # Evaluate the roll angle using a CNN
    pitch = my_head_pose_estimator.return_pitch(image, radians=True)  # Evaluate the pitch angle using a CNN
    yaw = my_head_pose_estimator.return_yaw(image, radians=True)  # Evaluate the yaw angle using a CNN
    print("Estimated [roll, pitch, yaw] (radians) ..... [" + str(roll[0,0,0]) + "," + str(pitch[0,0,0]) + "," + str(yaw[0,0,0])  + "]")
    #Getting rotation and translation vector
    rot_matrix = yaw2rotmat(-yaw[0,0,0]) #Deepgaze use different convention for the Yaw, we have to use the minus sign

    #Attention: OpenCV uses a right-handed coordinates system:
    #Looking along optical axis of the camera, X goes right, Y goes downward and Z goes forward.
    rvec, jacobian = cv2.Rodrigues(rot_matrix)
    tvec = np.array([0.0, 0.0, 1.0], np.float) # translation vector
    print rvec

    imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)
    p_start = (int(c_x), int(c_y))
    p_stop = (int(imgpts[2][0][0]), int(imgpts[2][0][1]))
    print("point start: " + str(p_start))
    print("point stop: " + str(p_stop))
    print("")

    cv2.line(image, p_start, p_stop, (0,0,255), 3) #RED
    cv2.circle(image, p_start, 1, (0,255,0), 3) #GREEN

    cv2.imwrite(file_save, image)


