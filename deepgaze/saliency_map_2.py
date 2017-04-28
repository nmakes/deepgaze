#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2017 Massimiliano Patacchiola
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import cv2
import threading
from timeit import default_timer as timer

class FasaSaliencyMapping:
    """Implementation of the FASA (Fast, Accurate, and Size-Aware Salient Object Detection) algorithm.

    Abstract:
    Fast and accurate salient-object detectors are important for various image processing and computer vision 
    applications, such as adaptive compression and object segmentation. It is also desirable to have a detector that is 
    aware of the position and the size of the salient objects. In this paper, we propose a salient-object detection 
    method that is fast, accurate, and size-aware. For efficient computation, we quantize the image colors and estimate 
    the spatial positions and sizes of the quantized colors. We then feed these values into a statistical model to 
    obtain a probability of saliency. In order to estimate the final saliency, this probability is combined with a 
    global color contrast measure. We test our method on two public datasets and show that our method significantly 
    outperforms the fast state-of-the-art methods. In addition, it has comparable performance and is an order of 
    magnitude faster than the accurate state-of-the-art methods. We exhibit the potential of our algorithm by 
    processing a high-definition video in real time. 
    """

    def __init__(self, image_h, image_w):
        """Init the classifier.

        """
        # Assigning some gloabl variables and creating here the image to fill later (for speed purposes)
        self.image_rows = image_h
        self.image_cols = image_w
        self.salient_image = np.zeros((image_h, image_w), dtype=np.uint8)
        # mu: mean vector
        self.mean_vector = np.array([0.5555, 0.6449, 0.0002, 0.0063])
        # covariance matrix
        self.covariance_matrix = np.array([[0.0231, -0.0010, 0.0001, -0.0002],
                                           [-0.0010, 0.0246, -0.0000, 0.0000],
                                           [0.0001, -0.0000, 0.0115, 0.0003],
                                           [-0.0002, 0.0000, 0.0003, 0.0080]])
        # determinant of covariance matrix
        # self.determinant_covariance = np.linalg.det(self.covariance_matrix)
        self.determinant_covariance = 5.21232874e-08

        # calculate the inverse of the covariance matrix
        self.covariance_matrix_inverse = np.array([[43.3777, 1.7633, -0.4059, 1.0997],
                                                   [1.7633, 40.7221, -0.0165, 0.0447],
                                                   [-0.4059, -0.0165, 87.0455, -3.2744],
                                                   [1.0997, 0.0447, -3.2744, 125.1503]])

    def _calculate_histogram(self, image, tot_bins=8):
        # 1- Conversion from BGR to LAB color space
        # Here a color space conversion is done. Moreover the min/max value for each channel is found.
        # This is helpful because the 3D histogram will be defined in this sub-space.
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        minL, maxL, _, _ = cv2.minMaxLoc(image[:, :, 0])
        minA, maxA, _, _ = cv2.minMaxLoc(image[:, :, 1])
        minB, maxB, _, _ = cv2.minMaxLoc(image[:, :, 2])

        # 2- Histograms in a 3D manifold of shape (tot_bin, tot_bin, tot_bin).
        # The cv2.calcHist for a 3-channels image generates a cube of size (tot_bins, tot_bins, tot_bins) which is a
        # discretization of the 3-D space defined by hist_range.
        # E.G. if range is 0-255 and it is divided in 5 bins we get -> [0-50][50-100][100-150][150-200][200-250]
        # So if you access the histogram with the indeces: histogram[3,0,2] it is possible to see how many pixels
        # fall in the range channel_1=[150-200], channel_2=[0-50], channel_3=[100-150]
        # data = np.vstack((image[:, :, 0].flat, image[:, :, 1].flat, image[:, :, 2].flat)).astype(np.uint8).T

        self.L_range = np.linspace(minL, maxL, num=tot_bins, endpoint=True)
        self.A_range = np.linspace(minA, maxA, num=tot_bins, endpoint=True)
        self.B_range = np.linspace(minB, maxB, num=tot_bins, endpoint=True)
        self.L_id_matrix = np.digitize(image[:, :, 0], self.L_range, right=True)
        self.A_id_matrix = np.digitize(image[:, :, 1], self.A_range, right=True)
        self.B_id_matrix = np.digitize(image[:, :, 2], self.B_range, right=True)
        # Here I compute the histogram manually, this allow saving time because during the image
        # inspection it is possible to allocate other useful information
        self.histogram = np.zeros((tot_bins, tot_bins, tot_bins))
        # it maps the 3D index of hist in a flat 1D array index
        self.map_3d_1d = np.zeros((tot_bins, tot_bins, tot_bins), dtype=np.int32)
        # this matrix contains for each bin: mx, my, mx^2, my^2
        self.centx_matrix  = np.zeros((tot_bins, tot_bins, tot_bins))  # mx
        self.centy_matrix  = np.zeros((tot_bins, tot_bins, tot_bins))  # my
        self.centx2_matrix = np.zeros((tot_bins, tot_bins, tot_bins))  # mx^2
        self.centy2_matrix = np.zeros((tot_bins, tot_bins, tot_bins))  # my^2

        for y in xrange(0, self.image_rows):
            for x in xrange(0, self.image_cols):
                L_id = self.L_id_matrix[y,x]
                A_id = self.A_id_matrix[y,x]
                B_id = self.B_id_matrix[y,x]
                self.centx_matrix[L_id, A_id, B_id] += x + 1e-10
                self.centy_matrix[L_id, A_id, B_id] += y + 1e-10
                self.centx2_matrix[L_id, A_id, B_id] += x * x + 1e-10  # np.power(x, 2)
                self.centy2_matrix[L_id, A_id, B_id] += y * y + 1e-10  # np.power(y, 2)
                self.histogram[L_id, A_id, B_id] += 1
        return image

    # 2- Like in the cpp code. Returns: map, colorDistance [matrix], exponentialColorDistance [matrix]
    # the dimensions of colorDistance and exponentialColorDistance is shape (tot_bins, tot_bins)
    def _precompute_parameters(self, sigmac=16):
        # This line creates a 3D cube containing the coordinates of the centroids.
        # Using these indices it is possible to find the closest centroid to an image pixel.
        L_centroid, A_centroid, B_centroid = np.meshgrid(self.L_range, self.A_range, self.B_range)
        # It gets the indeces of the values with non-zero bin in the histogram 3D matrix
        # this save iteration time because skip bins with empty values
        self.index_matrix = np.transpose(np.nonzero(self.histogram))
        self.number_of_colors = np.amax(self.index_matrix.shape)
        self.color_distance_matrix = np.zeros((self.number_of_colors, self.number_of_colors))
        self.exponential_color_distance_matrix = np.identity(self.number_of_colors)
        # Iterates on the indeces
        for i in xrange(0, self.number_of_colors):
            #self.color_distance_matrix[i, i] = 0.0
            #self.exponential_color_distance_matrix[i, i] = 1.0
            i_index = self.index_matrix[i, :]
            L_i = L_centroid[i_index[0], i_index[1], i_index[2]]
            A_i = A_centroid[i_index[0], i_index[1], i_index[2]]
            B_i = B_centroid[i_index[0], i_index[1], i_index[2]]
            i_vector = np.array([L_i, A_i, B_i])
            self.map_3d_1d[i_index[0], i_index[1], i_index[2]] = i  # the map is assigned here for performance purposes
            for k in xrange(i + 1, self.number_of_colors):
                k_index = self.index_matrix[k, :]
                L_k = L_centroid[k_index[0], k_index[1], k_index[2]]
                A_k = A_centroid[k_index[0], k_index[1], k_index[2]]
                B_k = B_centroid[k_index[0], k_index[1], k_index[2]]
                k_vector = np.array([L_k, A_k, B_k])
                color_difference = np.sum(np.power(i_vector-k_vector, 2))
                # color_difference = np.power(L_i - L_k, 2) + np.power(A_i - A_k, 2) + np.power(B_i - B_k, 2)
                self.color_distance_matrix[i, k] = np.sqrt(color_difference)
                self.color_distance_matrix[k, i] = self.color_distance_matrix[i, k]
                self.exponential_color_distance_matrix[i, k] = np.exp(- color_difference / (2 * sigmac * sigmac))
                self.exponential_color_distance_matrix[k, i] = self.exponential_color_distance_matrix[i, k]
        return self.number_of_colors

    def _bilateral_filtering(self):
        """ Applying the bilateral filtering to the matrices.
        
        Since the trick 'matrix[ matrix > x]' is used it would be possible to set a threshold
        which is an energy value, considering only the histograms which have enough colours.
        @return: mx, my, Vx, Vy
        """
        # Obtaining the values through vectorized operations (very efficient)
        self.contrast = np.dot(self.color_distance_matrix, self.histogram[self.histogram > 0])
        normalization_array = np.dot(self.exponential_color_distance_matrix, self.histogram[self.histogram > 0])
        self.mx = np.dot(self.exponential_color_distance_matrix, self.centx_matrix[self.centx_matrix > 0])
        self.my = np.dot(self.exponential_color_distance_matrix, self.centy_matrix[self.centy_matrix > 0])
        mx2 = np.dot(self.exponential_color_distance_matrix, self.centx2_matrix[self.centx2_matrix > 0])
        my2 = np.dot(self.exponential_color_distance_matrix, self.centy2_matrix[self.centy2_matrix > 0])
        # Normalizing the vectors
        self.mx = np.divide(self.mx, normalization_array)
        self.my = np.divide(self.my, normalization_array)
        mx2 = np.divide(mx2, normalization_array)
        my2 = np.divide(my2, normalization_array)
        self.Vx = np.subtract(mx2, np.power(self.mx, 2))
        self.Vy = np.subtract(my2, np.power(self.my, 2))
        return self.mx, self.my, self.Vx, self.Vy

    def _calculate_probability(self):
        """ Vectorized version of the probability estimation.
        
        :return: a vector shape_probability of shape (number_of_colors)
        """
        g = np.array([np.sqrt(12 * self.Vx) / self.image_cols,
                      np.sqrt(12 * self.Vy) / self.image_rows,
                      (self.mx - (self.image_cols / 2.0)) / float(self.image_cols),
                      (self.my - (self.image_rows / 2.0)) / float(self.image_rows)])
        X = (g.T - self.mean_vector)
        Y = X
        A = self.covariance_matrix_inverse
        result = (np.dot(X, A) * Y).sum(1)  # This line does the trick
        self.shape_probability = np.exp(- result / 2)

    def _compute_saliency_map(self):
        self.saliency = np.multiply(self.contrast, self.shape_probability)
        for i in xrange(0, self.number_of_colors):
            a1 = 0
            a2 = 0
            for k in xrange(0, self.number_of_colors):
                if self.exponential_color_distance_matrix[i,k] > 0:
                    a1 += self.saliency[k] * self.exponential_color_distance_matrix[i,k]
                    a2 += self.exponential_color_distance_matrix[i,k]
            self.saliency[i] = a1/a2

        # the saliency vector is renormalised in range [0-255]
        minVal, maxVal, _, _ = cv2.minMaxLoc(self.saliency)
        self.saliency = self.saliency - minVal
        self.saliency = 255 * self.saliency / (maxVal - minVal) + 1e-3
        return self.saliency

    def return_contrast_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        contrast = self.contrast
        minVal, maxVal, _, _ = cv2.minMaxLoc(contrast)
        contrast = contrast - minVal
        contrast = 255 * contrast / (maxVal - minVal) + 1e-3
        image_salient = np.zeros((self.image_rows, self.image_cols))
        for y in xrange(0, self.image_rows):
            for x in xrange(0, self.image_cols):
                L_id = int(np.digitize(image[y, x, 0], self.L_range, right=True))
                A_id = int(np.digitize(image[y, x, 1], self.A_range, right=True))
                B_id = int(np.digitize(image[y, x, 2], self.B_range, right=True))
                index = np.argmax(np.all(self.index_matrix == [L_id, A_id, B_id], axis=1))
                image_salient[y,x] = contrast[index]
        return np.uint8(image_salient)

    def return_probability_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        probability = self.shape_probability
        minVal, maxVal, _, _ = cv2.minMaxLoc(probability)
        probability = probability - minVal
        probability = 255 * probability / (maxVal - minVal) + 1e-3
        image_salient = np.zeros((self.image_rows, self.image_cols))
        for y in xrange(0, self.image_rows):
            for x in xrange(0, self.image_cols):
                L_id = int(np.digitize(image[y, x, 0], self.L_range, right=True))
                A_id = int(np.digitize(image[y, x, 1], self.A_range, right=True))
                B_id = int(np.digitize(image[y, x, 2], self.B_range, right=True))
                index = np.argmax(np.all(self.index_matrix == [L_id, A_id, B_id], axis=1))
                image_salient[y,x] = probability[index]
        return np.uint8(image_salient)

    def returnMask(self, image, tot_bins=8, format='BGR2LAB'):
        if format == 'BGR2LAB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif format == 'BGR2RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif format == 'RGB2LAB':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        elif format == 'RGB' or format == 'BGR' or format == 'LAB':
            pass
        else:
            raise ValueError('[DEEPGAZE][SALIENCY MAP][ERROR] the input format of the image is not supported.')
        start = timer()
        self._calculate_histogram(image, tot_bins=tot_bins)
        end = timer()
        print("--- %s calculate_histogram seconds ---" % (end - start))
        start = timer()
        number_of_colors = self._precompute_parameters()
        end = timer()
        print("--- number of colors: " + str(number_of_colors) + " ---")
        print("--- %s precompute_paramters seconds ---" % (end - start))
        start = timer()
        self._bilateral_filtering()
        end = timer()
        print("--- %s bilateral_filtering seconds ---" % (end - start))
        start = timer()
        self._calculate_probability()
        end = timer()
        print("--- %s calculate_probability seconds ---" % (end - start))
        start = timer()
        self._compute_saliency_map()
        end = timer()
        print("--- %s compute_saliency_map seconds ---" % (end - start))
        '''
        for y in xrange(0, self.image_rows):
            for x in xrange(0, self.image_cols):
                L_id = int(np.digitize(image[y, x, 0], self.L_range, right=True))
                A_id = int(np.digitize(image[y, x, 1], self.A_range, right=True))
                B_id = int(np.digitize(image[y, x, 2], self.B_range, right=True))
                index = np.argmax(np.all(self.index_matrix == [L_id, A_id, B_id], axis=1))
                self.salient_image[y,x] = self.saliency[index]
        '''
        # Obtain the index of the image single pixel
        #self.L_id_matrix = np.digitize(image[:, :, 0], self.L_range, right=True)
        #self.A_id_matrix = np.digitize(image[:, :, 1], self.A_range, right=True)
        #self.B_id_matrix = np.digitize(image[:, :, 2], self.B_range, right=True)
        #index_list = self.index_matrix.tolist()
        start = timer()
        it = np.nditer(self.salient_image, flags=['multi_index'], op_flags=['writeonly'])
        while not it.finished:
            y = it.multi_index[0]
            x = it.multi_index[1]
            L_id = self.L_id_matrix[y, x]
            A_id = self.A_id_matrix[y, x]
            B_id = self.B_id_matrix[y, x]
            index = self.map_3d_1d[L_id, A_id, B_id]
            it[0] = self.saliency[index]
            it.iternext()
        end = timer()
        print("--- %s returnMask 'iteration part' seconds ---" % (end - start))
        return self.salient_image


def main():

    image_path = "/home/massimiliano/Desktop/fasa_images/horse.jpg"
    image = cv2.imread(image_path)
    my_map = FasaSaliencyMapping(image.shape[0], image.shape[1])

    start = timer()
    image = cv2.imread(image_path)
    image_salient = my_map.returnMask(image, tot_bins=8, format='BGR2LAB')
    end = timer()
    print("--- %s Tot seconds ---" % (end - start))

    image_contrast = my_map.return_contrast_image(image)
    image_probability = my_map.return_probability_image(image)
    #print ("Number of colours: " + str(number_of_colours))

    cv2.imshow("Original", image)
    #cv2.imshow("Contrast", image_contrast)
    cv2.imshow("Saliency Map", image_salient)
    #cv2.imshow("Probability", image_probability)


    while True:
        if cv2.waitKey(33) == ord('q'):
            cv2.destroyAllWindows()
            break


def main_webcam():
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 320)
    video_capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 180)
    print video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    print video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

    if(video_capture.isOpened() == False):
        print("Error: the resource is busy or unvailable")
        return
    else:
        print("The video source has been opened correctly...")

    #Create the main window and move it
    cv2.namedWindow('Video')
    cv2.moveWindow('Video', 20, 20)

    #Obtaining the CAM dimension
    cam_w = int(video_capture.get(3))
    cam_h = int(video_capture.get(4))

    my_map = FasaSaliencyMapping(cam_h, cam_w)

    while(True):

        # Capture frame-by-frame
        ret, frame = video_capture.read()
        image_salient = my_map.returnMask(frame, tot_bins=8, format='BGR2LAB')

        cv2.imshow('Video', image_salient)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
