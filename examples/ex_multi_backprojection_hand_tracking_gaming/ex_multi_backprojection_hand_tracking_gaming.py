#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#In this example I used the backprojection algorithm with multimple templates
# in order to track my hand in a webcam streaming. The center of the hand is
#taken as reference point for controlling some keys on the keyboard and play
#a videogame. To obtain the templates of your hands you can simply take some
#screenshot of the open hand using some screen capture utilities (like Shutter)
#You can load as many templates as you like, just remember to load the images
#and append them in the template_list. To simulate the keyboard I am using the
#libraray evdev that requires admin right in order to write in your keyboard.
#To run the example just type: sudo python ex_multi_backprojection_hand_tracking_gaming.py

#BUTTONS:
# 'a' = Press 'a' to start the capture of the hand position and the
# keyboard simulation
# 'q' = Press 'q' to exit (you have to select the CV windows with the mouse)

import cv2
import numpy as np
from evdev import UInput, ecodes as e
from deepgaze.color_detection import MultiBackProjectionColorDetector
from deepgaze.mask_analysis import BinaryMaskAnalyser

#Declare the simulated keyboard object
ui = UInput()
#Enable or disavle the keyboard simulation (enabled when press 'a')
ENABLE_CAPTURE = False

#Declare a list and load the templates. If you are using more templates
#then you have to load them here.
template_list=list()
template_list.append(cv2.imread('template_1.png')) #Load the image
template_list.append(cv2.imread('template_2.png')) #Load the image
template_list.append(cv2.imread('template_3.png')) #Load the image
template_list.append(cv2.imread('template_4.png')) #Load the image
template_list.append(cv2.imread('template_5.png')) #Load the image
template_list.append(cv2.imread('template_6.png')) #Load the image

#Open a webcam streaming
video_capture=cv2.VideoCapture(0) #Open the webcam
#Reduce the size of the frame to 320x240
video_capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 320)
video_capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 240)
#Get the webcam resolution
cam_w = int(video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
cam_h = int(video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
#Declare an offset that is used to define the distance
#from the webcam center of the two red lines
offset = int(cam_h / 7)

#Declaring the binary mask analyser object
my_mask_analyser = BinaryMaskAnalyser()

#Defining the deepgaze color detector object
my_back_detector = MultiBackProjectionColorDetector()
my_back_detector.setTemplateList(template_list) #Set the template 

print("Welcome! Press 'a' to start the hand tracking. Press 'q' to exit...")

while(True):
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if(frame is None): break #check for empty frames
    #Return the binary mask from the backprojection algorithm
    frame_mask = my_back_detector.returnMask(frame, morph_opening=True, blur=True, kernel_size=5, iterations=2)
    if(my_mask_analyser.returnNumberOfContours(frame_mask) > 0 and ENABLE_CAPTURE==True):
        x_center, y_center = my_mask_analyser.returnMaxAreaCenter(frame_mask)
        x_rect, y_rect, w_rect, h_rect = my_mask_analyser.returnMaxAreaRectangle(frame_mask)
        area = w_rect * h_rect
        cv2.circle(frame, (x_center, y_center), 3, [0,255,0], 5)
        #Check the position of the target and press the keys
        #KEY_UP, KEY_DOWN, KEY_RIGHT, KEY_LEFT, KEY_SPACE
        #KEY_W, KEY_S, KEY_D, KEY_A
        #DOWN
        if(y_center > int(cam_h/2)+offset and area>10000):
            ui.write(e.EV_KEY, e.KEY_DOWN, 1)
            print("KEY_DOWN")     
        #UP
        elif(y_center < int(cam_h/2)-offset and area>10000):
            ui.write(e.EV_KEY, e.KEY_UP, 1)
            print("KEY_UP")
        else:
            print("WAITING") 
            ui.write(e.EV_KEY, e.KEY_DOWN, 0) #release the buttons
            ui.write(e.EV_KEY, e.KEY_UP, 0)          
        ui.syn()

    #Drawing the offsets
    cv2.line(frame, (0, int(cam_h/2)-offset), (cam_w, int(cam_h/2)-offset), [0,0,255], 2) #horizontal
    cv2.line(frame, (0, int(cam_h/2)+offset), (cam_w, int(cam_h/2)+offset), [0,0,255], 2)

    #Showing the frame and waiting for the exit command
    cv2.imshow('mpatacchiola - deepgaze', frame) #show on window
    cv2.imshow('Mask', frame_mask) #show on window
    if cv2.waitKey(1) & 0xFF == ord('q'): break #Exit when Q is pressed
    if cv2.waitKey(33) == ord('a'): 
        if(ENABLE_CAPTURE==True): 
            print("Disabling capture...")
            ENABLE_CAPTURE=False
        else:
            print("Enabling capture...") 
            ENABLE_CAPTURE=True

#Close the keyboard ojbect
ui.close()
#Release the camera
video_capture.release()
print("Bye...")
