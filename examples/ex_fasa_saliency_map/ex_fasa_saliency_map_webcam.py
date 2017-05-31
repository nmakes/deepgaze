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

# In this example the FASA algorithm is used in order to process the webcam stream.

import numpy as np
import cv2
from timeit import default_timer as timer
from deepgaze.saliency_map import FasaSaliencyMapping 

# If True it prints the time for processing the frame.
PRINT_TIME = False

# Using OpenCV the resolution of the webcam is set to these values.
# You must check which resolution your webcam support and adjust the values in accordance.
RESOLUTION_WIDTH = 320
RESOLUTION_HEIGHT = 180

def main():
    # Open the video stream and set the webcam resolution.
    # It may give problem if your webcam does not support the particular resolution used.
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, RESOLUTION_WIDTH)
    video_capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, RESOLUTION_HEIGHT)
    print video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    print video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

    if(video_capture.isOpened() == False):
        print("Error: the resource is busy or unvailable")
        return
    else:
        print("The video source has been opened correctly...")

    # Create the main window and move it
    cv2.namedWindow('Video')
    cv2.moveWindow('Video', 20, 20)

    # Obtaining the CAM dimension
    cam_w = int(video_capture.get(3))
    cam_h = int(video_capture.get(4))

    # Defining the FASA object using the camera resolution
    my_map = FasaSaliencyMapping(cam_h, cam_w)

    while True:
        start = timer()
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        image_salient = my_map.returnMask(frame, tot_bins=8, format='BGR2LAB')
        end = timer()
        # Print the time for processing the frame
        if PRINT_TIME: 
            print("--- %s Tot seconds ---" % (end - start))
            print("")
        cv2.imshow('Video', image_salient)
        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
