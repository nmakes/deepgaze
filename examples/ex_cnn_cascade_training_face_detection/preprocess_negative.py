#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2017 Massimiliano Patacchiola
# https://mpatacchiola.github.io
# https://mpatacchiola.github.io/blog/
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import numpy as np
from six.moves import cPickle as pickle
import cv2
import os


def main():
    img_size = 12  # size to resize the image to
    image_list = list()
    dataset_path = "./detection/neg_patches"
    counter = 1

    # Iterates the negative images and resize them
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):
            print("Image number ..... " + str(counter))
            print("Image name   ..... " + str(filename))
            print("")
            image_path = os.path.join(dataset_path, filename)
            image = cv2.imread(image_path)
            image_rescaled = cv2.resize(image, (img_size,img_size), interpolation = cv2.INTER_AREA)
            image_list.append(image_rescaled)
            counter += 1

    # Creating the dataset
    tot_images = counter
    training_label = np.zeros((tot_images, 2))
    training_label[:,1] = 1
    training_dataset = np.asarray(image_list)


    pickle_file = "./negative_dataset_"  + str(img_size) + "net_" + str(tot_images) + ".pickle"
    print("Saving the dataset in: " + pickle_file)
    print("... ")

    try:
             print("Opening the file...")
             f = open(pickle_file, 'wb')
             save = {
               'training_dataset': training_dataset,
               'training_label': training_label   
                   }

             print("Training dataset: ", training_dataset.shape)
             print("Training label: ", training_label.shape)
             print("Saving the file...")
             pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
             print("Closing the file...")
             f.close()
             print("")
             print("The dataset has been saved and it is ready for the training! \n")
             print("")
    except Exception as e:
             print('Unable to save data to', pickle_file, ':', e)
             raise

if __name__ == "__main__":
    main()
