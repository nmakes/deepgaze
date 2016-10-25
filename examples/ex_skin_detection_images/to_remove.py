import numpy as np
import cv2
from deepgaze.color_detection import RangeColorDetector

min_range = np.array([0, 58, 50], dtype = "uint8") #lower HSV boundary of skin color
max_range = np.array([30, 255, 255], dtype = "uint8") #upper HSV boundary of skin color
my_skin_detector = RangeColorDetector(min_range, max_range) #Define the detector object

image = cv2.imread("tomb_rider.jpg") #Read the image with OpenCV
image_filtered = my_skin_detector.returnFiltered(image, morph_opening=False, blur=False)
image_stack = np.hstack((image, image_filtered))
cv2.imshow('image', image_stack)
cv2.waitKey(0)
cv2.destroyAllWindow()
