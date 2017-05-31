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

# In this example the FASA algorithm is used in order to process some images.
# The original image and the saliency version are showed for comparison.

import numpy as np
import cv2
from timeit import default_timer as timer
from deepgaze.saliency_map import FasaSaliencyMapping 

def main():

    image_1 = cv2.imread("./horse.jpg")
    image_2 = cv2.imread("./car.jpg")
    image_3 = cv2.imread("./plane.jpg")
    image_4 = cv2.imread("./pear.jpg")

    # for each image the same operations are repeated
    my_map = FasaSaliencyMapping(image_1.shape[0], image_1.shape[1])  # init the saliency object
    start = timer()
    image_salient_1 = my_map.returnMask(image_1, tot_bins=8, format='BGR2LAB')  # get the mask from the original image
    image_salient_1 = cv2.GaussianBlur(image_salient_1, (3,3), 1)  # applying gaussin blur to make it pretty
    end = timer()
    print("--- %s Image 1 tot seconds ---" % (end - start))

    my_map = FasaSaliencyMapping(image_2.shape[0], image_2.shape[1])
    start = timer()
    image_salient_2 = my_map.returnMask(image_2, tot_bins=8, format='BGR2LAB')
    image_salient_2 = cv2.GaussianBlur(image_salient_2, (3,3), 1)
    end = timer()
    print("--- %s Image 2 tot seconds ---" % (end - start))

    my_map = FasaSaliencyMapping(image_3.shape[0], image_3.shape[1])
    start = timer()
    image_salient_3 = my_map.returnMask(image_3, tot_bins=8, format='BGR2LAB')
    #image_salient_3 = cv2.GaussianBlur(image_salient_3, (3,3), 1)
    end = timer()
    print("--- %s Image 3 tot seconds ---" % (end - start))

    my_map = FasaSaliencyMapping(image_4.shape[0], image_4.shape[1])
    start = timer()
    image_salient_4 = my_map.returnMask(image_4, tot_bins=8, format='BGR2LAB')
    image_salient_4 = cv2.GaussianBlur(image_salient_4, (3,3), 1)
    end = timer()
    print("--- %s Image 4 tot seconds ---" % (end - start))

    # Creating stack of images and showing them on screen
    original_images_stack = np.hstack((image_1, image_2, image_3, image_4))
    saliency_images_stack = np.hstack((image_salient_1, image_salient_2, image_salient_3, image_salient_4))
    saliency_images_stack = np.dstack((saliency_images_stack,saliency_images_stack,saliency_images_stack))
    cv2.imshow("Original-Saliency", np.vstack((original_images_stack, saliency_images_stack)))

    while True:
        if cv2.waitKey(33) == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
