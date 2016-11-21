#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#In this example I plot the result of the intersection between 
#two histograms. The histogram intersection has been introduced
#by Ballard and Swain in their article "Color Indexing".

import cv2
import numpy as np
from matplotlib import pyplot as plt

#Formula for estimating the intersection in
#two one-dimensional histograms
def histogram_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection

def main():

    bins_number = 100
    #Changing the mean value you can manage to get the two
    #histograms. If mu_1==mu_2 then the intersection is close 
    #to 1 more the two values are different more the intersection
    #is close to zero.
    mu_1 = -2
    mu_2 = 2
    data_1 = np.random.normal(mu_1, 2.0, 1000) #RED
    data_2 = np.random.normal(mu_2, 2.0, 1000) #GREEN
    hist_1, bin_edges_1 = np.histogram(data_1, bins=bins_number, range=[-15, 15])
    hist_2, bin_edges_2 = np.histogram(data_2, bins=bins_number, range=[-15, 15])

    #Find the intersection and print the value
    intersection = histogram_intersection(hist_1, hist_2)
    print("Intersection Value: " + str(intersection))

    #Display the graph using matplotlib
    font_size = 20
    plt.bar(np.arange(bins_number), hist_1, 1, color='r', alpha=0.2)
    plt.bar(np.arange(bins_number), hist_2, 1, color='b', alpha=0.2)
    plt.title("Histogram Intersection", size=font_size)
    plt.xlabel('bins', size=font_size)
    plt.ylabel('number of elements', size=font_size)
    plt.show()


if __name__ == "__main__": 
    main()
