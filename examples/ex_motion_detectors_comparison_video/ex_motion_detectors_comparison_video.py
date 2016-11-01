#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#In this example three motion detectors are compared:
#frame differencing, MOG, MOG2.
#Given a video as input "cars.avi" it returns four
#different videos: original, differencing, MOG, MOG2

import numpy as np
import cv2
from deepgaze.motion_detection import DiffMotionDetector
from deepgaze.motion_detection import MogMotionDetector
from deepgaze.motion_detection import Mog2MotionDetector

#Open the video file and loading the background image
video_capture = cv2.VideoCapture("./cars.avi")
background_image = cv2.imread("./background.png")

#Decalring the diff motion detector object and setting the background
my_diff_detector = DiffMotionDetector()
my_diff_detector.setBackground(background_image)
#Declaring the MOG motion detector
my_mog_detector = MogMotionDetector()
my_mog_detector.returnMask(background_image)
#Declaring the MOG 2 motion detector
my_mog2_detector = Mog2MotionDetector()
my_mog2_detector.returnGreyscaleMask(background_image)

# Define the codec and create VideoWriter objects
fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter("./cars_original.avi", fourcc, 20.0, (1920,1080))
out_diff = cv2.VideoWriter("./cars_diff.avi", fourcc, 20.0, (1920,1080))
out_mog = cv2.VideoWriter("./cars_mog.avi", fourcc, 20.0, (1920,1080))
out_mog2 = cv2.VideoWriter("./cars_mog2.avi", fourcc, 20.0, (1920,1080))


while(True):

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    #Get the mask from the detector objects
    diff_mask = my_diff_detector.returnMask(frame)
    mog_mask = my_mog_detector.returnMask(frame)
    mog2_mask = my_mog2_detector.returnGreyscaleMask(frame)

    #Merge the b/w frame in order to have depth=3
    diff_mask = cv2.merge([diff_mask, diff_mask, diff_mask])
    mog_mask = cv2.merge([mog_mask, mog_mask, mog_mask])
    mog2_mask = cv2.merge([mog2_mask, mog2_mask, mog2_mask])

    #Writing in the output file
    out.write(frame)
    out_diff.write(diff_mask)
    out_mog.write(mog_mask)
    out_mog2.write(mog2_mask)

    #Showing the frame and waiting
    # for the exit command
    if(frame is None): break #check for empty frames
    cv2.imshow('Original', frame) #show on window
    cv2.imshow('Diff', diff_mask) #show on window
    cv2.imshow('MOG', mog_mask) #show on window
    cv2.imshow('MOG 2', mog2_mask) #show on window
    if cv2.waitKey(1) & 0xFF == ord('q'): break #Exit when Q is pressed

#Release the camera
video_capture.release()
print("Bye...")
