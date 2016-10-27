#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import cv2
from deepgaze.motion_detection import DiffMotionDetector
from deepgaze.mask_analysis import BinaryMaskAnalyser

#Open the video file and loading the background image
video_capture = cv2.VideoCapture("./cars.avi")
background_image = cv2.imread("./background.png")

#Decalring the motion detector object and setting the background
my_motion_detector = DiffMotionDetector()
my_motion_detector.setBackground(background_image)

#Declaring the binary mask analyser object
my_mask_analyser = BinaryMaskAnalyser()

# Define the codec and create VideoWriter object
fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter("./cars_deepgaze.avi", fourcc, 20.0, (1920,1080))

#Create the main window and move it
cv2.namedWindow('Video')
cv2.moveWindow('Video', 20, 20)
is_first_frame = True

while(True):

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    frame_mask = my_motion_detector.returnMask(frame)
    #cx, cy = my_mask_analyser.returnMaxAreaCenter(frame_mask)
    #cnt = my_mask_analyser.returnMaxAreaContour(frame_mask)
    
    if(my_mask_analyser.returnNumberOfContours(frame_mask) > 0):
        x,y,w,h = my_mask_analyser.returnMaxAreaRectangle(frame_mask)
        cv2.rectangle(frame, (x,y), (x+w,y+h), [0,255,0], 2)

    #Writing in the output file
    out.write(frame)

    #Showing the frame and waiting
    # for the exit command
    if(frame is None): break #check for empty frames
    cv2.imshow('Video', frame) #show on window
    if cv2.waitKey(1) & 0xFF == ord('q'): break #Exit when Q is pressed

#Release the camera
video_capture.release()
print("Bye...")
