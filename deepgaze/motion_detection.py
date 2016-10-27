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

class DiffMotionDetector:
    """Motion is detected through the difference between 
       the background (static) and the foregroung (dynamic).

    This class calculated the absolute difference between two frames.
    The first one is a static frame which represent the background 
    and the second is the image containing the moving object.
    The resulting mask is passed to a threshold and cleaned from noise. 
    """

    def __init__(self):
        """Init the color detector object.

    """
        self.background_gray = None

    def setBackground(self, frame):
        """Set the BGR image used as template during the pixel selection
 
        The template can be a spedific region of interest of the main
        frame or a representative color scheme to identify. the template
        is internally stored as an HSV image.
        @param frame the template to use in the algorithm
        """
        if(frame is None): return None 
        self.background_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def getBackground(self):
        """Get the BGR image used as template during the pixel selection
 
        The template can be a spedific region of interest of the main
        frame or a representative color scheme to identify.
        """
        if(self.background_gray is None): 
            return None
        else:
            return cv2.cvtColor(self.background_gray, cv2.COLOR_GRAY2BGR)

    def returnMask(self, foreground_image, threshold=25):
        if(foreground_image is None): return None
        foreground_gray = cv2.cvtColor(foreground_image, cv2.COLOR_BGR2GRAY)
        delta_image = cv2.absdiff(self.background_gray, foreground_gray)
	threshold_image = cv2.threshold(delta_image, threshold, 255, cv2.THRESH_BINARY)[1]
        return threshold_image



         
