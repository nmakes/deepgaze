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

#def histogram_intersection(image, model):
#    I = np.histogram(image)
#    M = np.histogram(model)
#    minima = np.minimum(I, M)
#    intersection = np.sum(minima) / numpy.sum(M)
#    return intersection

def histogram_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection

def main():

    bins_number = 100
    model_1 = cv2.imread('model_3.png', 0)
    #rng = np.random.RandomState(10)  # deterministic random data
    #a = np.hstack((rng.normal(size=1000), rng.normal(loc=5, scale=2, size=1000)))
    mu_1 = 0
    mu_2 = 0
    data_1 = np.random.normal(mu_1, 2.0, 1000) #RED
    data_2 = np.random.normal(mu_2, 2.0, 1000) #GREEN
    hist_1, bin_edges_1 = np.histogram(data_1, bins=bins_number, range=[-15, 15])
    hist_2, bin_edges_2 = np.histogram(data_2, bins=bins_number, range=[-15, 15])

    intersection = histogram_intersection(hist_1, hist_2)

    print(hist_1)
    print(hist_2)
    print(intersection)


    #plt.fill(np.arange(bins_number), hist_1, 'Crimson')
    plt.bar(np.arange(bins_number), hist_1, 1, color='r', alpha=0.2)
    plt.bar(np.arange(bins_number), hist_2, 1, color='b', alpha=0.2)
    #plt.hist(hist, bins=bins_number)  # plt.hist passes it's arguments to np.histogram
    plt.title("Histogram Intersection", size=35)
    plt.xlabel('bins', size=35)
    plt.ylabel('number of elements', size=35)
    plt.show()


if __name__ == "__main__": main()
