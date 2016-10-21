#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import cv2
import numpy as np
from deepgaze.color_detection import BackProjectionColorDetector

image = cv2.imread('tiger.jpg') #Load the image
template = image[225:275,625:675] #Taking a subframe of the image
my_back_detector = BackProjectionColorDetector()#Defining the deepgaze color detector object
my_back_detector.setTemplate(template) #Set the template 
image_filtered = my_back_detector.returnFiltered(image, morph_opening=True, blur=True, kernel_size=7, iterations=2)

cv2.rectangle(image, (625, 225), (675, 275), (0,255,0), 2) #Drawing a green rectangle around the template
images_stack = np.hstack((image,image_filtered)) #The images are stack in order
#cv2.imwrite("tiger_filtered.jpg", images_stack) #Save the image if you prefer
cv2.imshow('image', images_stack) #Show the images on screen
cv2.waitKey(0)
cv2.destroyAllWindows()
