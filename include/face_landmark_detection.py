#!/usr/bin/env python

## @package face_landmark_detection.py
#
# Massimiliano Patacchiola, Plymouth University 2016
#
# This module requires dlib >= 18.10 because of the use
# of the shape predictor object.

import numpy
import sys
import cv2
import dlib
import os.path


class faceLandmarkDetection:


    def __init__(self, landmarkPath):
        #Check if the file provided exist
        if(os.path.isfile(landmarkPath)==False):
            raise ValueError('haarCascade: the files specified do not exist.') 

        self._predictor = dlib.shape_predictor(landmarkPath)

    ##
    # Find landmarks in the image provided
    # @param inputImg the image where the algorithm will be called
    #
    def returnLandmarks(self, inputImg):
            #The predictor will be run on the whole image
            #Then it is necessary to reshape it on the face
            inputImg_w, inputImg_h = inputImg.shape[::-1]
            size = cv2.cv.GetSize(inputImg)
            print(size)
            #Creating a dlib rectangle and finding the landmarks
            dlib_rectangle = dlib.rectangle(left=0, top=0, right=int(inputImg_w), bottom=int(inputImg_h))
            landmarks = self._predictor(inputImg, dlib_rectangle)
            #It creates a numpy.matrix containing x-y coordinates of the 68 landmarks found by the predictor
            self._landmarks_matrix = numpy.matrix([[p.x, p.y] for p in landmarks.parts()])
            return self._landmarks_matrix


    ##
    # Find landmarks in the image provided
    # @param inputImg the image where the algorithm will be called
    #
    def returnLandmarks(self, inputImg, roiX, roiY, roiW, roiH):
            #Creating a dlib rectangle and finding the landmarks
            dlib_rectangle = dlib.rectangle(left=int(roiX), top=int(roiY), right=int(roiW), bottom=int(roiH))
            landmarks = self._predictor(inputImg, dlib_rectangle)
            #It creates a numpy.matrix containing x-y coordinates of the 68 landmarks found by the predictor
            self._landmarks_matrix = numpy.matrix([[p.x, p.y] for p in landmarks.parts()])
            return self._landmarks_matrix






