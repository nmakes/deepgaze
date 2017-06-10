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
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--img_size', required=True,
                        help='The size of the images: 12, 24, 48')
    parser.add_argument('-i', '--input_directory', required=True,
                        help='The directory containing the images')
    args = vars(parser.parse_args())

    img_size = int(args['img_size'])  # size to resize the image to
    image_list = list()
    dataset_path = args['input_directory'] # "./detection/pos_faces"
    counter = 1

    if not os.path.exists(dataset_path):
        print("[DEEPGAZE][ERROR]: The specified folder does not exist: " + str(dataset_path))
        return

    for root, dirs, files in os.walk(dataset_path):
        for filename in files:
            if filename.endswith(".jpg"):
                image_path = os.path.join(root, filename)
                image = cv2.imread(image_path)
                image_dimension = image.shape[0]
                if image_dimension >= img_size:
                    print("Image number    ..... " + str(counter))
                    print("Image name      ..... " + str(filename))
                    print("Image dimension ..... " + str(image_dimension))
                    print("")
                    image_rescaled = cv2.resize(image, (img_size,img_size), interpolation = cv2.INTER_AREA)
                    image_list.append(image_rescaled)
                    counter += 1
                else:
                    print("Image rejected!")
                    print("Image name      ..... " + str(filename))
                    print("Image dimension ..... " + str(image_dimension))
                    print("")

    # Creating the dataset
    tot_images = counter
    training_label = np.zeros((tot_images, 2))
    training_label[:,0] = 1
    training_dataset = np.asarray(image_list)

    # Store in pickle
    pickle_file = "./positive_dataset_" + str(img_size) + "net_" + str(tot_images) + ".pickle"
    print("Saving the dataset in: " + pickle_file)
    print("... ")
    try:
        print("Opening the file...")
        f = open(pickle_file, 'wb')
        save = {'training_dataset': training_dataset,
                'training_label': training_label}

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
