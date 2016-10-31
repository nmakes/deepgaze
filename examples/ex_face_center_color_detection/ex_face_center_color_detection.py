#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# In this example I will use the range color detector class to find a face in a picture.
# To find a face it is possible to filter the skin color and then take the center
# of the contour with the largest area. In most of the case this contour is the face.
# This example can be extended using face detector in order to check if the contour
# is a face or something else.

import numpy as np
import cv2
from deepgaze.color_detection import RangeColorDetector
from deepgaze.mask_analysis import BinaryMaskAnalyser

#Declaring the boundaries and creating the object
min_range = np.array([0, 58, 50], dtype = "uint8") #lower HSV boundary of skin color
max_range = np.array([30, 255, 255], dtype = "uint8") #upper HSV boundary of skin color
my_skin_detector = RangeColorDetector(min_range, max_range) #Define the detector object
my_mask_analyser = BinaryMaskAnalyser()

image = cv2.imread("tomb_rider_2.jpg") #Read the image with OpenCV
#For this image we use one iteration of the morph_opening and gaussian blur to clear the noise
image_filtered = my_skin_detector.returnFiltered(image, morph_opening=True, blur=True, kernel_size=3, iterations=1)
#To use the function returnMaxAreaCenter we need to have the balck and white mask
image_mask = my_skin_detector.returnMask(image, morph_opening=True, blur=True, kernel_size=3, iterations=1) 

#Here we get the center of the contour with largest area and
#the contour points.
cx, cy = my_mask_analyser.returnMaxAreaCenter(image_mask)
cnt = my_mask_analyser.returnMaxAreaContour(image_mask)

#Uncomment if you want to get the coords of the rectangle sorrounding
#the largest area contour (in this case the face)
#x, y, w, h = my_mask_analyser.returnMaxAreaRectangle(image_mask)

#Uncomment if you want to get the coords of the circle sorrounding
#the largest area contour (in this case the face)
#(x, y), radius = my_mask_analyser.returnMaxAreaCircle(image_mask)

#Drawing and displaying
image_canvas = np.copy(image)
#Uncomment the line below only if you uncomment also the line
#with returnMaxAreaRectangle(image_mask) function.
#cv2.rectangle(image_canvas, (x,y), (x+w,y+h), [255,0,0], 3)
#cv2.circle(image_canvas, (x,y), radius, (255,0,0), 3)
cv2.drawContours(image_canvas, [cnt], 0, (0,255,0), 2)
cv2.circle(image_canvas, (cx, cy), 2, [0, 0,255], 3)
image_stack = np.hstack((image, image_filtered, image_canvas))
#Uncomment if you want to save the image
#cv2.imwrite("tomb_rider_2_filtered.jpg", image_filtered) #Save the filtered image
cv2.imshow("image", image_stack)
cv2.waitKey(0)
cv2.destroyAllWindows()
