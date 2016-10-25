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
from deepgaze.color_detection import RangeColorDetector

min_range = np.array([0, 58, 50], dtype = "uint8") #lower HSV boundary of skin color
max_range = np.array([30, 255, 255], dtype = "uint8") #upper HSV boundary of skin color
my_skin_detector = RangeColorDetector(min_range, max_range) #Define the detector object


image = cv2.imread("tomb_rider_2.jpg") #Read the image with OpenCV
my_skin_detector.setRange(min_range, max_range) #Set the new range for the color detector object
#For this image we use one iteration of the morph_opening and gaussian blur to clear the noise
image_filtered = my_skin_detector.returnFiltered(image, morph_opening=True, blur=True, kernel_size=3, iterations=1)

image_mask = my_skin_detector.returnMask(image, morph_opening=True, blur=True, kernel_size=3, iterations=1) 

cx, cy = my_skin_detector.returnMaxAreaCenter(image_mask)
cnt = my_skin_detector.returnMaxAreaContour(image_mask)
image_canvas = np.copy(image)
cv2.drawContours(image_canvas, [cnt], 0, (0,255,0), 2)
cv2.circle(image_canvas, (cx, cy), 2, [0, 0,255], 3)
image_stack = np.hstack((image, image_filtered, image_canvas))
#cv2.imshow("image", image_stack)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

cv2.imwrite("tomb_rider_2_detection.jpg", image_stack) #Save the filtered image
