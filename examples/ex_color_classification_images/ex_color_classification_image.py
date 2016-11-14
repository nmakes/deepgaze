#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#In this example the Backprojection algorithm is used in order to find the pixels that have the same
#HSV histogram of a predefined template. The template is a subframe of the main image or an external
#matrix that can be used as a filter. In this example I take a subframe of the main image (the tiger
# fur) and I use it for obtaining a filtered version of the original frame. A green rectangle shows
#where the subframe is located.

import cv2
import numpy as np
from matplotlib import pyplot as plt
from deepgaze.color_classification import HistogramColorClassifier

my_classifier = HistogramColorClassifier(channels=[0, 1, 2], hist_size=[10, 10, 10], bins_range=[0, 256, 0, 256, 0, 256])

model_1 = cv2.imread('model_1_c.jpg')
model_2 = cv2.imread('model_2_c.jpg')
model_3 = cv2.imread('model_3_c.jpg')
model_4 = cv2.imread('model_4_c.jpg')
model_5 = cv2.imread('model_5_c.jpg')
model_6 = cv2.imread('model_6_c.jpg')
model_7 = cv2.imread('model_7_c.jpg')
model_8 = cv2.imread('model_8_c.jpg')

my_classifier.addModelHistogram(model_1)
my_classifier.addModelHistogram(model_2)
my_classifier.addModelHistogram(model_3)
my_classifier.addModelHistogram(model_4)
my_classifier.addModelHistogram(model_5)
my_classifier.addModelHistogram(model_6)
my_classifier.addModelHistogram(model_7)
my_classifier.addModelHistogram(model_8)

image = cv2.imread('model_1.jpg') #Load the image
comparison_array = my_classifier.returnHistogramComparisonArray(image, method="correlation")


#model_hist = cv2.calcHist([model_1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#model_hist = cv2.normalize(model_hist).flatten()
#image_hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#image_hist = cv2.normalize(image_hist).flatten()
#comparison = my_classifier.returnHistogramComparison(model_hist, image_hist, method='intersection')

#image_hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#image_hist = cv2.normalize(image_hist).flatten()
#plt.hist(image_hist.ravel(),256,[0,256]); plt.show()

#model_hist = cv2.calcHist([model_1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#model_hist = cv2.normalize(model_hist).flatten()
#comp = cv2.compareHist(image_hist, model_hist, cv2.cv.CV_COMP_INTERSECT)
#print(comp)

print(comparison_array)













