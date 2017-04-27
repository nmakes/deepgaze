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

    def __init__(self):
        """Init the classifier.

        """
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

    def _calculate_histogram(self, image, tot_bins=8, quantize_image=False):
        # 1- Conversion from BGR to LAB color space
        # Here a color space conversion is done. Moreover the min/max value for each channel is found.
        # This is helpful because the 3D histogram will be defined in this sub-space.
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        self.image_rows = image.shape[0]
        self.image_cols = image.shape[1]
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

        # 3- This line creates a 3D cube containing the coordinates of the centroids.
        # Using these indeces it is possible to find the closest centroid to an image pixel.
        self.L_range = np.linspace(minL, maxL, num=tot_bins, endpoint=True)
        self.A_range = np.linspace(minA, maxA, num=tot_bins, endpoint=True)
        self.B_range = np.linspace(minB, maxB, num=tot_bins, endpoint=True)
        self.L_centroid, self.A_centroid, self.B_centroid = np.meshgrid(self.L_range, self.A_range, self.B_range)

        # Here I compute the histogram manually, this allow saving time because during the image
        # inspection it is possible to allocate other useful information
        self.histogram = np.zeros((tot_bins, tot_bins, tot_bins))
        # this matrix contains for each bin: mx, my, mx^2, my^2
        self.centvar_matrix = np.zeros((tot_bins, tot_bins, tot_bins, 4))
        for y in xrange(0, self.image_rows):
            for x in xrange(0, self.image_cols):
                L_id = int(np.digitize(image[y, x, 0], self.L_range, right=True))
                A_id = int(np.digitize(image[y, x, 1], self.A_range, right=True))
                B_id = int(np.digitize(image[y, x, 2], self.B_range, right=True))
                # L = self.L_centroid[L_id, A_id, B_id]
                # A = self.A_centroid[L_id, A_id, B_id]
                # B = self.B_centroid[L_id, A_id, B_id]
                # image[y, x, :] = [L, A, B]
                self.centvar_matrix[L_id, A_id, B_id, 0] += x
                self.centvar_matrix[L_id, A_id, B_id, 1] += y
                self.centvar_matrix[L_id, A_id, B_id, 2] += np.power(x, 2)
                self.centvar_matrix[L_id, A_id, B_id, 3] += np.power(y, 2)
                self.histogram[L_id, A_id, B_id] += 1
        return image

    # 2- Like in the cpp code. Returns: map, colorDistance [matrix], exponentialColorDistance [matrix]
    # the dimensions of colorDistance and exponentialColorDistance is shape (tot_bins, tot_bins)
    # TAKES: Mat histogram, vector<float> LL, vector<float> AA, vector<float> BB, int numberOfPixels, vector<int> &reverseMap, Mat &map, Mat &colorDistance, Mat &exponentialColorDistance
    # PASSED: histogram, LL, AA, BB, im.cols * im.rows, reverseMap, map, colorDistance, exponentialColorDistance
    def _precompute_paramters(self, sigmac=16):
        # It gets the indeces of the values with non-zero bin in the histogram 3D matrix
        # this save iteration time because skip bins with empty values
        index_matrix = np.transpose(np.nonzero(self.histogram))
        number_of_colors = np.amax(index_matrix.shape)
        self.color_distance_matrix = np.zeros((number_of_colors, number_of_colors))
        self.exponential_color_distance_matrix = np.zeros((number_of_colors, number_of_colors))
        # Iterates on the indeces
        for i in xrange(0, number_of_colors):
            self.color_distance_matrix[i, i] = 0.0
            self.exponential_color_distance_matrix[i, i] = 1.0
            i_index = index_matrix[i, :]
            L_i = self.L_centroid[i_index[0], i_index[1], i_index[2]]
            A_i = self.A_centroid[i_index[0], i_index[1], i_index[2]]
            B_i = self.B_centroid[i_index[0], i_index[1], i_index[2]]
            for k in xrange(i + 1, number_of_colors):
                k_index = index_matrix[k, :]
                L_k = self.L_centroid[k_index[0], k_index[1], k_index[2]]
                A_k = self.A_centroid[k_index[0], k_index[1], k_index[2]]
                B_k = self.B_centroid[k_index[0], k_index[1], k_index[2]]
                color_difference = np.power(L_i - L_k, 2) + np.power(A_i - A_k, 2) + np.power(B_i - B_k, 2)
                self.color_distance_matrix[i, k] = np.sqrt(color_difference)
                self.color_distance_matrix[k, i] = np.sqrt(color_difference)
                self.exponential_color_distance_matrix[i, k] = np.exp(- color_difference / (2 * sigmac * sigmac))
                self.exponential_color_distance_matrix[k, i] = np.exp(- color_difference / (2 * sigmac * sigmac))
        return number_of_colors

    def _bilateral_filtering(self):
        index_matrix = np.transpose(np.nonzero(self.histogram))
        number_of_colors = np.amax(index_matrix.shape)
        self.contrast = np.zeros(number_of_colors)
        self.mx = np.zeros(number_of_colors)
        self.my = np.zeros(number_of_colors)
        mx2 = np.zeros(number_of_colors)
        my2 = np.zeros(number_of_colors)
        normalization_array = np.zeros(number_of_colors)
        for i in xrange(0, number_of_colors):
            i_index = index_matrix[i, :]
            # L_i = self.L_centroid[i_index[0], i_index[1], i_index[2]]
            # A_i = self.A_centroid[i_index[0], i_index[1], i_index[2]]
            # B_i = self.B_centroid[i_index[0], i_index[1], i_index[2]]
            for k in xrange(0, number_of_colors):
                k_index = index_matrix[k, :]
                # L_k = self.L_centroid[k_index[0], k_index[1], k_index[2]]
                # A_k = self.A_centroid[k_index[0], k_index[1], k_index[2]]
                # B_k = self.B_centroid[k_index[0], k_index[1], k_index[2]]
                # Here the main arrays are calculated
                self.contrast[i] += self.color_distance_matrix[i, k] * self.histogram[
                    k_index[0], k_index[1], k_index[2]]
                self.mx[i] += self.exponential_color_distance_matrix[i, k] * self.centvar_matrix[
                    k_index[0], k_index[1], k_index[2], 0]
                self.my[i] += self.exponential_color_distance_matrix[i, k] * self.centvar_matrix[
                    k_index[0], k_index[1], k_index[2], 1]
                mx2[i] += self.exponential_color_distance_matrix[i, k] * self.centvar_matrix[
                    k_index[0], k_index[1], k_index[2], 2]
                my2[i] += self.exponential_color_distance_matrix[i, k] * self.centvar_matrix[
                    k_index[0], k_index[1], k_index[2], 3]
                normalization_array[i] += self.exponential_color_distance_matrix[i, k] * self.histogram[
                    k_index[0], k_index[1], k_index[2]]

        self.mx = np.divide(self.mx, normalization_array)
        self.my = np.divide(self.my, normalization_array)
        mx2 = np.divide(mx2, normalization_array)
        my2 = np.divide(my2, normalization_array)
        self.Vx = np.subtract(mx2, np.power(self.mx, 2))
        self.Vy = np.subtract(my2, np.power(self.my, 2))
        return self.mx, self.my, self.Vx, self.Vy

    def _calculate_probability(self):
        index_matrix = np.transpose(np.nonzero(self.histogram))
        number_of_colors = np.amax(index_matrix.shape)
        self.shape_probability = np.zeros(number_of_colors)
        # Caclulating the g vector
        for k in xrange(0, number_of_colors):
            # Calculate the g vectors for the saliency probability
            g = np.array([np.sqrt(12 * self.Vx[k]) / self.image_cols,
                          np.sqrt(12 * self.Vy[k]) / self.image_rows,
                          (self.mx[k] - (self.image_cols / 2.0)) / float(self.image_cols),
                          (self.my[k] - (self.image_rows / 2.0)) / float(self.image_rows) ])
            # Allocate the result in the vector
            result = np.dot(self.covariance_matrix_inverse, g - self.mean_vector)
            result = np.dot(result, g - self.mean_vector)
            self.shape_probability[k] = np.exp(- result / 2)
            #self.shape_probability[k] = (1 / ((2 * np.pi) ** 2 * np.sqrt(self.determinant_covariance))) * \
            #                            np.exp(-np.dot((g - self.mean_vector), np.dot(self.covariance_matrix_inverse,
            #                                                                          g - self.mean_vector)) / 2)
        return self.shape_probability

    def _compute_saliency_map(self):
        number_of_colors = np.amax(self.contrast.shape)
        self.saliency = np.multiply(self.contrast, self.shape_probability)
        for i in xrange(0, number_of_colors):
            a1 = 0
            a2 = 0
            for k in xrange(0, number_of_colors):
                if self.exponential_color_distance_matrix[i,k] > 0:
                    a1 += self.saliency[k] * self.exponential_color_distance_matrix[i,k]
                    a2 += self.exponential_color_distance_matrix[i,k]
            self.saliency[i] = a1/a2

        # the saliency vector is renormalised in range [0-255]
        minVal, maxVal, _, _ = cv2.minMaxLoc(self.saliency)
        self.saliency = self.saliency - minVal
        self.saliency = 255 * self.saliency / (maxVal - minVal) + 1e-3
        return self.saliency

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
        self._calculate_histogram(image, tot_bins=tot_bins)
        self._precompute_paramters()
        self._bilateral_filtering()
        self._calculate_probability()
        self._compute_saliency_map()
        image_salient = np.zeros((self.image_rows, self.image_cols))
        index_matrix = np.transpose(np.nonzero(self.histogram))
        for y in xrange(0, self.image_rows):
            for x in xrange(0, self.image_cols):
                L_id = int(np.digitize(image[y, x, 0], self.L_range, right=True))
                A_id = int(np.digitize(image[y, x, 1], self.A_range, right=True))
                B_id = int(np.digitize(image[y, x, 2], self.B_range, right=True))
                index = np.argmax(np.all(index_matrix == [L_id, A_id, B_id], axis=1))
                image_salient[y,x] = self.saliency[index]
        return np.uint8(image_salient)

    def return_contrast_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        contrast = self.contrast
        minVal, maxVal, _, _ = cv2.minMaxLoc(contrast)
        contrast = contrast - minVal
        contrast = 255 * contrast / (maxVal - minVal) + 1e-3
        image_salient = np.zeros((self.image_rows, self.image_cols))
        index_matrix = np.transpose(np.nonzero(self.histogram))
        for y in xrange(0, self.image_rows):
            for x in xrange(0, self.image_cols):
                L_id = int(np.digitize(image[y, x, 0], self.L_range, right=True))
                A_id = int(np.digitize(image[y, x, 1], self.A_range, right=True))
                B_id = int(np.digitize(image[y, x, 2], self.B_range, right=True))
                index = np.argmax(np.all(index_matrix == [L_id, A_id, B_id], axis=1))
                image_salient[y,x] = contrast[index]
        return np.uint8(image_salient)

    def return_probability_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        probability = self.shape_probability
        minVal, maxVal, _, _ = cv2.minMaxLoc(probability)
        probability = probability - minVal
        probability = 255 * probability / (maxVal - minVal) + 1e-3
        image_salient = np.zeros((self.image_rows, self.image_cols))
        index_matrix = np.transpose(np.nonzero(self.histogram))
        for y in xrange(0, self.image_rows):
            for x in xrange(0, self.image_cols):
                L_id = int(np.digitize(image[y, x, 0], self.L_range, right=True))
                A_id = int(np.digitize(image[y, x, 1], self.A_range, right=True))
                B_id = int(np.digitize(image[y, x, 2], self.B_range, right=True))
                index = np.argmax(np.all(index_matrix == [L_id, A_id, B_id], axis=1))
                image_salient[y,x] = probability[index]
        return np.uint8(image_salient)


def main():
    import time
    start_time = time.time()
    my_map = FasaSaliencyMapping()
    image = cv2.imread("/home/massimiliano/Desktop/fasa_images/horse.jpg")
    image_salient = my_map.returnMask(image, tot_bins=8, format='BGR2LAB')
    print("--- %s seconds ---" % (time.time() - start_time))

    #image_quantized = my_map._calculate_histogram(image, tot_bins=8)
    #number_of_colours = my_map._precompute_paramters()
    #mx, my, Vx, Vy = my_map._bilateral_filtering()
    #probability = my_map._calculate_probability()
    #saliency = my_map._compute_saliency_map()
    #image_salient = my_map.return_saliency_map(image)
    image_contrast = my_map.return_contrast_image(image)
    image_probability = my_map.return_probability_image(image)
    #print ("Number of colours: " + str(number_of_colours))

    cv2.imshow("Original", image)
    cv2.imshow("Contrast", image_contrast)
    cv2.imshow("Saliency Map", image_salient)
    cv2.imshow("Probability", image_probability)
    # cv2.imshow("Color Quantization BGR", cv2.cvtColor(np.uint8(image_quantized), cv2.COLOR_LAB2BGR))
    # cv2.imshow("lab", image_lab)
    while True:
        if cv2.waitKey(33) == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
