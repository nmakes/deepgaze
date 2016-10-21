#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import cv2
import sys

class BackProjectionColorDetector:
    """Implementation of the Histogram Backprojection algorithm.

    The histogram backprojection was proposed by Michael Swain and Dana Ballard 
    in their paper "Indexing via color histograms".
    Abstract: The color spectrum of multicolored objects provides a a robust, 
    efficient cue for indexing into a large database of models. This paper shows 
    color histograms to be stable object representations over change in view, and 
    demonstrates they can differentiate among a large number of objects. It introduces 
    a technique called Histogram Intersection for matching model and image histograms 
    and a fast incremental version of Histogram Intersection that allows real-time 
    indexing into a large database of stored models using standard vision hardware. 
    Color can also be used to search for the location of an object. An algorithm 
    called Histogram Backprojection performs this task efficiently in crowded scenes.
    """

    def __init__(self):
        """Init the color detector object.

    """
        self.template_hsv = None

    def setTemplate(self, frame):
        """Set the BGR image used as template during the pixel selection
 
        The template can be a spedific region of interest of the main
        frame or a representative color scheme to identify. the template
        is internally stored as an HSV image.
        @param frame the template to use in the algorithm
        """      
        self.template_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def getTemplate(self):
        """Get the BGR image used as template during the pixel selection
 
        The template can be a spedific region of interest of the main
        frame or a representative color scheme to identify.
        """
        if(self.template_hsv == None): 
            return None
        else:
            return cv2.cvtColor(self.template_hsv, cv2.COLOR_HSV2BGR)

    def returnFiltered(self, frame):
        """Given an input frame in BGR return the filtered version.
 
        @param frame the original frame (color)
        """
        if(self.template_hsv == None): return None
        #Convert the input framge from BGR -> HSV
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #Set the template histogram
        template_hist = cv2.calcHist([self.template_hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
        #Normalize the template histogram and apply backprojection
        cv2.normalize(template_hist, template_hist, 0, 255, cv2.NORM_MINMAX)
        frame_hsv = cv2.calcBackProject([frame_hsv], [0,1], template_hist, [0,180,0,256], 1)
        #Get the kernel and apply a convolution
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        frame_hsv = cv2.filter2D(frame_hsv, -1, kernel)
        #Get the threshold
        ret, frame_threshold = cv2.threshold(frame_hsv, 50, 255, 0)
        #Merge the threshold matrices
        frame_threshold = cv2.merge((frame_threshold,frame_threshold,frame_threshold))
        #Return the AND image
        return cv2.bitwise_and(frame, frame_threshold)


class RangeColorDetector:
    """Using this detector it is possible to isolate colors in a specified range.

    In this detector the frame given as input is filtered and the pixel which
    fall in a specific range are taken, the other rejected. Some erosion and
    dilatation operation are used in order to remove noise.
    This class use the HSV (Hue, Saturation, Value) color representation to filter pixels.
    The H and S components characterize the color (independent of illumination) 
    and V compoenent specifies the illuminations.
    """

    def __init__(self, min_range, max_range):
        """Init the color detector object.

        The object must be initialised with an HSV range to use as filter.
        Ex: skin color in channel H is characterized by values between [0, 20], 
        in the channel S=[48, 255] and V=[80, 255] (Asian and Caucasian). To
        initialise the vectors in this range it is possible to write:       
        min_range = numpy.array([0, 48, 80], dtype = "uint8")
        max_range = numpy.array([20, 255, 255], dtype = "uint8")
        @param range_min the minimum HSV value to use as filer (numpy.array)
        @param range_max the maximum HSV value to use as filter (numpy.array)
        """
        # min and max range to use as filter for the detector (HSV)
        self.min_range = min_range
        self.max_range = max_range

    def setRange(self, min_range, max_range):
        """Set the min and max range used in the range detector
 
        The skin in channel H is characterized by values between 0 and 50, 
        in the channel S from 0.23 to 0.68 (Asian and Caucasian).
        @param range_min the minimum HSV value to use as filer
        @param range_max the maximum HSV value to use as filter
        """
        # min and max range to use as filter for the detector (HSV)
        self.min_range = min_range
        self.max_range = max_range

    def getRange(self):
        """Return the min and max range used in the skin detector
 
        """
        return (self.min_range, self.max_range)


    def returnFiltered(self, frame):
        """Given an input frame return the filtered version.
 
        @param frame the original frame (color)
        """
        #Convert to HSV and eliminate pixels outside the range
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_filtered = cv2.inRange(frame_hsv, self.min_range, self.max_range)
        #Bitwise mask
	frame_filtered = cv2.bitwise_and(frame, frame, mask = frame_filtered)
        return frame_filtered

    def returnRawFiltered(self, frame):
        """Given an input frame return the Raw filtered version.
 
        This version of the function does not use the blur and bitwise 
        operations, then the resulting frame contains white pixels
        in correspondance of the skin found during the searching process.
        @param frame the original frame (color)
        """
        #Convert to HSV and eliminate pixels outside the range
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_filtered = cv2.inRange(frame_hsv, self.min_range, self.max_range)
        return frame_filtered


    def returnFilteredDenoised(self, frame, iterations=2, blur=True):
        """Given an input frame return the filtered and denoised version.
 
        @param frame the original frame (color)
        @param iterations the number of time erode and dilate are called
        @param blur if True it uses a Gaussian Kernel to remove Gaussian noise
        """
        #Convert to HSV and eliminate pixels outside the range
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_filtered = cv2.inRange(frame_hsv, self.min_range, self.max_range)
        #Applying some denoising operation on the frame
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
	frame_filtered = cv2.erode(frame_filtered, kernel, iterations = iterations)
	frame_filtered = cv2.dilate(frame_filtered, kernel, iterations = iterations)
        #Blur and then bitwise mask
	if(blur==True): frame_filtered = cv2.GaussianBlur(frame_filtered, (3, 3), 0)
	frame_denoised = cv2.bitwise_and(frame, frame, mask = frame_filtered)
        return frame_denoised







