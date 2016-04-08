#!/usr/bin/env python

import os.path
import numpy
import cv2
import csv
import matplotlib.pyplot as plt
#import tensorflow as tf
from sklearn.preprocessing import normalize


#If the dataset do not exist it create it
if(os.path.isfile("label.npy")==False or os.path.isfile("dataset.npy")==False):

    #Saving the file names in a list
    image_list = list()
    with open("../tugraz_dataset/aflw/data/label.csv", 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            image_list.append(row[0])

    #Loading the label
    label = numpy.genfromtxt("../tugraz_dataset/aflw/data/label.csv", delimiter=',', skip_header=0, usecols=(range(1,4)), dtype=numpy.float32)
    label_row, label_col = label.shape
    print(label.shape)
    print(label[5][:])

    dataset_row = label_row
    dataset_col = 40 * 40 #the size of the image
    dataset = numpy.zeros((dataset_row, dataset_col), dtype=numpy.int8)

    row_counter = 0
    for image_name in image_list:
        image_path = "../tugraz_dataset/aflw/data/output/" + image_name
        image = cv2.imread(image_path, 0) #load in greyscale
        dataset[row_counter] = image.reshape(1, -1)
        row_counter += 1 


    #Saving the numpy array to files
    numpy.save("label", label)
    numpy.save("dataset", dataset)


#Load the dataset and labels
label = numpy.load("label.npy")
dataset = numpy.load("dataset.npy")

print(dataset[8][0:100])

data = normalize(label[:,0]).ravel()
plt.hist(data, bins=300)
plt.title("Roll Histogram")
#plt.xlim(-4,4)
#plt.ylim(0,1450)
plt.xlabel("Radians")
plt.ylabel("Frequency")
plt.show()


