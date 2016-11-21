#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#In this example the histogram intersection algorithm is used in order
#to classify eight different superheroes. The histogram intersection is
#one of the simplest classifier and it uses histograms as 
#comparison to identify the best match between an input image and a model.

import cv2
import numpy as np
from matplotlib import pyplot as plt
from deepgaze.color_classification import HistogramColorClassifier

#Defining the classifier
my_classifier = HistogramColorClassifier(channels=[0, 1, 2], hist_size=[128, 128, 128], hist_range=[0, 256, 0, 256, 0, 256], hist_type='BGR')

model_1 = cv2.imread('model_1.png') #Flash
model_2 = cv2.imread('model_2.png') #Batman
model_3 = cv2.imread('model_3.png') #Hulk
model_4 = cv2.imread('model_4.png') #Superman
model_5 = cv2.imread('model_5.png') #Capt. America
model_6 = cv2.imread('model_6.png') #Wonder Woman
model_7 = cv2.imread('model_7.png') #Iron Man
model_8 = cv2.imread('model_8.png') #Wolverine
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
#Get a numpy array which contains the comparison values
#between the model and the input image
comparison_array = my_classifier.returnHistogramComparisonArray(image, method="intersection")
#Normalisation of the array
comparison_distribution = comparison_array / np.sum(comparison_array)

#Printing the arrays
print("Comparison Array:")
print(comparison_array)
print("Distribution Array: ")
print(comparison_distribution)

#Plotting a bar chart with the probability distribution
#If you are comparing more than 8 superheroes you have to
#change the total objects variable and add new labels in 
total_objects = 8
label_objects = ('Flash', 'Batman', 'Hulk', 'Superman', 'Capt. America', 'Wonder Woman', 'Iron Man', 'Wolverine')
font_size = 20
width = 0.5 
plt.barh(np.arange(total_objects), comparison_distribution, width, color='r')
plt.yticks(np.arange(total_objects) + width/2.,label_objects , rotation=0, size=font_size)
plt.xlim(0.0, 1.0)
plt.ylim(-0.5, 8.0)
plt.xlabel('Probability', size=font_size)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()
