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

        #The feature points important
        # for the 3D tracking
        self._side_right = (0, 0) #0
        self._menton =  (0, 0) #8
        self._side_left = (0, 0) #16
        self._sellion = (0, 0) #27
        self._nose = (0, 0) #30
        #self._philtrum = (0, 0) #51



    ##
    # Find landmarks in the image provided
    # @param inputImg the image where the algorithm will be called
    #
    def returnLandmarks(self, inputImg, roiX, roiY, roiW, roiH):
            #Creating a dlib rectangle and finding the landmarks
            dlib_rectangle = dlib.rectangle(left=int(roiX), top=int(roiY), right=int(roiW), bottom=int(roiH))
            landmarks = self._predictor(inputImg, dlib_rectangle)
            #It creates a numpy.matrix containing x-y coordinates of the 68 landmarks found by the predictor
            self._landmark_matrix = numpy.matrix([[p.x, p.y] for p in landmarks.parts()])

            self._nose = ( self._landmark_matrix[30].item((0,0)), self._landmark_matrix[30].item((0,1)) )
            self._sellion = ( self._landmark_matrix[27].item((0,0)), self._landmark_matrix[27].item((0,1)) )
            self._menton = self._landmark_matrix[8]
            #self._philtrum = self._landmark_matrix[51]
            self._side_left = self._landmark_matrix[16] 
            self._side_right = self._landmark_matrix[0]


            #These points are the one that have a correspondece
            # in the 3D space, because their data have been found
            # from antropometric tables. They will be used by solvePnP().
            self.landmark_main_points = numpy.float32([[self._landmark_matrix[0].item((0,0)),  self._landmark_matrix[0].item((0,1)) ],
                                                       [self._landmark_matrix[8].item((0,0)),  self._landmark_matrix[8].item((0,1)) ], 
                                                       [self._landmark_matrix[16].item((0,0)), self._landmark_matrix[16].item((0,1))],
                                                       [self._landmark_matrix[27].item((0,0)), self._landmark_matrix[27].item((0,1))],
                                                       [self._landmark_matrix[30].item((0,0)), self._landmark_matrix[30].item((0,1))],
                                                       [self._landmark_matrix[33].item((0,0)), self._landmark_matrix[33].item((0,1))],
                                                       [self._landmark_matrix[36].item((0,0)), self._landmark_matrix[36].item((0,1))],
                                                       [self._landmark_matrix[39].item((0,0)), self._landmark_matrix[39].item((0,1))],
                                                       [self._landmark_matrix[42].item((0,0)), self._landmark_matrix[42].item((0,1))],
                                                       [self._landmark_matrix[45].item((0,0)), self._landmark_matrix[45].item((0,1))],
                                                       [self._landmark_matrix[62].item((0,0)), self._landmark_matrix[62].item((0,1))]])
            return self._landmark_matrix


    def print_landmark_coords(self):
        print("NOSE: ", self._nose)
        #print("SELLION: ", self._sellion)





