#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#In this example I use the range detector class to detect skin in two pictures
#The range detector find which pixels are included in a specific range.
#The hardest part is to find the correct boundaries for the range and tune
#the detector with the right morphing operation in order to have clean results
#and remove noise. The filter use HSV color representation (https://en.wikipedia.org/wiki/HSL_and_HSV)

import numpy as np
import cv2
from deepgaze.color_detection import RangeColorDetector

#Firs image boundaries
min_range = np.array([0, 48, 70], dtype = "uint8") #lower HSV boundary of skin color
max_range = np.array([20, 150, 255], dtype = "uint8") #upper HSV boundary of skin color
my_skin_detector = RangeColorDetector(min_range, max_range) #Define the detector object
image = cv2.imread("tomb_rider.jpg") #Read the image with OpenCV
#We do not need to remove noise from this image so morph_opening and blur are se to False
image_filtered = my_skin_detector.returnFiltered(image, morph_opening=False, blur=False, kernel_size=3, iterations=1)
cv2.imwrite("tomb_rider_filtered.jpg", image_filtered) #Save the filtered image

#Second image boundaries
min_range = np.array([0, 58, 50], dtype = "uint8") #lower HSV boundary of skin color
max_range = np.array([30, 255, 255], dtype = "uint8") #upper HSV boundary of skin color
image = cv2.imread("tomb_rider_2.jpg") #Read the image with OpenCV
my_skin_detector.setRange(min_range, max_range) #Set the new range for the color detector object
#For this image we use one iteration of the morph_opening and gaussian blur to clear the noise
image_filtered = my_skin_detector.returnFiltered(image, morph_opening=True, blur=True, kernel_size=3, iterations=1)
cv2.imwrite("tomb_rider_2_filtered.jpg", image_filtered) #Save the filtered image
