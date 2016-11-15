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

my_classifier = HistogramColorClassifier(channels=[0, 1, 2], hist_size=[128, 128, 128], hist_range=[0, 256, 0, 256, 0, 256], hist_type='BGR')

model_1 = cv2.imread('model_1a.png') #Flash
model_2 = cv2.imread('model_2a.png') #Batman
model_3 = cv2.imread('model_3a.png') #Hulk
model_4 = cv2.imread('model_4a.png') #Superman
model_5 = cv2.imread('model_5a.png') #Capt. America
model_6 = cv2.imread('model_6a.png') #Wonder Woman
model_7 = cv2.imread('model_7a.png') #Iron Man
model_8 = cv2.imread('model_8a.png') #Wolverine
#model_9 = cv2.imread('model_9_c.png') #Thor
#model_10 = cv2.imread('model_10_c.png') #Magneto

my_classifier.addModelHistogram(model_1)
my_classifier.addModelHistogram(model_2)
my_classifier.addModelHistogram(model_3)
my_classifier.addModelHistogram(model_4)
my_classifier.addModelHistogram(model_5)
my_classifier.addModelHistogram(model_6)
my_classifier.addModelHistogram(model_7)
my_classifier.addModelHistogram(model_8)
#my_classifier.addModelHistogram(model_9)
#my_classifier.addModelHistogram(model_10)

image = cv2.imread('image_2.jpg') #Load the image
comparison_array = my_classifier.returnHistogramComparisonArray(image, method="intersection")
comparison_distribution = my_classifier.returnHistogramComparisonProbability(image, method="intersection")

print(comparison_array)
print("Distribution: ")
print(comparison_distribution)

width = 0.5 
plt.barh(np.arange(8), comparison_distribution, width, color='r')
plt.yticks(np.arange(8) + width/2., ('Flash', 'Batman', 'Hulk', 'Superman', 'Capt. America', 'Wonder Woman', 'Iron Man', 'Wolverine'), rotation=0, size=25)
plt.xlim(0.0, 1.0)
plt.ylim(-0.5, 8.0)
plt.xlabel('Probability', size=25)


#ax = plt.axes()
#ax.arrow(0, 0, 0.5, 0.5, head_width=0.05, head_length=0.1, fc='k', ec='k')

# the histogram of the data
#n, bins, patches = plt.hist(comparison_distribution, bins=8, histtype='bar', facecolor='green', alpha=0.75)
#hist, bins = np.histogram(comparison_distribution, bins=8)
#labels = ['Frogs', 'Hogs', 'Bogs', 'Slogs', 'Frogs', 'Hogs', 'Bogs', 'Slogs']
#plt.xticks(range(1,9), labels, rotation='vertical')
#width = 0.7 * (bins[1] - bins[0])
#center = (bins[:-1] + bins[1:]) / 2
#plt.bar(center, hist, align='center', width=width)

plt.show()









