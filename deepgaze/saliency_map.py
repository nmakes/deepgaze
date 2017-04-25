#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2017 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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
        self.covariance_matrix = np.array([[ 0.0231, -0.0010,  0.0001, -0.0002],
                                           [-0.0010,  0.0246, -0.0000,  0.0000],
                                           [ 0.0001, -0.0000,  0.0115,  0.0003],
                                           [-0.0002,  0.0000,  0.0003,  0.0080]])
        # determinant of covariance matrix
        self.determinant_covariance = np.linalg.det(self.covariance_matrix)
        # calculate the inverse of the covariance matrix
        self.covariance_matrix_inverse = np.array([[43.3777,    1.7633,   -0.4059,    1.0997],
                                                   [1.7633,   40.7221,   -0.0165,    0.0447],
                                                   [-0.4059,   -0.0165,   87.0455,   -3.2744],
                                                   [1.0997,    0.0447,   -3.2744,  125.1503]])

    def _return_quantized_image(self, image, number_of_clusters=8):
        """Returns a quantized version of the image using k-mean clustering.
        
        @param image: the original image (BGR)
        @param number_of_clusters: the number of cluster to use
        @return: the quantized image
        """

        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        #image = cv2.cvtColor(image, cv2.COLOR_LBGR2LAB)
        maxL = np.amax(image[:, :, 0])
        minL = np.amin(image[:, :, 0])
        maxA = np.amax(image[:, :, 1])
        minA = np.amin(image[:, :, 1])
        maxB = np.amax(image[:, :, 2])
        minB = np.amin(image[:, :, 2])
        # Reshape the image and convert to float32
        Z = image.reshape((-1, 3))
        Z = np.float32(Z)
        # Define criteria = ( type, max_iter = 5 , epsilon = 1.0 )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 3, 1.0)
        # Set flags (Just to avoid line break in the code)
        flags = cv2.KMEANS_RANDOM_CENTERS
        # cv2.kmeans(data, K, criteria, attempts, flags[, bestLabels[, centers]]) -> retval, bestLabels, centers
        compactness, label, centers = cv2.kmeans(Z, number_of_clusters, criteria, 3, flags)
        # Count how many element in each bin
        bincount_array = np.bincount(label.flatten())
        # Back to uint8
        # center = np.uint8(center)
        res = np.uint8(centers)[label.flatten()]
        image_quantized = res.reshape((image.shape))
        pixel_to_index = np.vstack(np.arange(number_of_clusters))
        pixel_to_index = pixel_to_index[label.flatten()]
        return image, image_quantized, centers, pixel_to_index[:,0].reshape((image.shape[0],image.shape[1])), bincount_array

    def _return_centers_variances(self, cluster_centers_array, cluster_labels_array, bincount_array):
        K = np.amax(cluster_centers_array.shape)
        m_xk = np.zeros(K)
        m_yk = np.zeros(K)
        V_xk = np.zeros(K)
        V_yk = np.zeros(K)
        for k in range(K):
            w_c = np.zeros(K)
            y_i_list = list()
            x_i_list = list()
            Q_k = cluster_centers_array[k]
            for j in range(K):
                Q_j = cluster_centers_array[j]
                w_c[j] = self._return_color_weights(Q_k, Q_j)
                x_i_list.append(np.where(cluster_labels_array == j)[1])  # taking the column indeces
                y_i_list.append(np.where(cluster_labels_array == j)[0])  # taking the rows indeces
            # To be faster it estimates the denominator only once
            denominator = np.sum(np.multiply(bincount_array, w_c))
            # Estimate the center
            m_xk[k] = np.sum(w_c * np.sum(x_i_list[k])) / denominator
            m_yk[k] = np.sum(w_c * np.sum(y_i_list[k])) / denominator
            # Estimate the variance
            V_xk[k] = np.sum(w_c * np.sum((x_i_list[k] - m_xk[k])**2)) / denominator
            V_yk[k] = np.sum(w_c * np.sum((y_i_list[k] - m_yk[k])**2)) / denominator
        return m_xk, m_yk, V_xk, V_yk, w_c


    def _return_centers_variances(self, cluster_centers_array, cluster_labels_array, bincount_array):
        K = np.amax(cluster_centers_array.shape)
        m_xk = np.zeros(K)
        m_yk = np.zeros(K)
        V_xk = np.zeros(K)
        V_yk = np.zeros(K)
        for k in range(K):
            w_c = np.zeros(K)
            y_i_list = list()
            x_i_list = list()
            Q_k = cluster_centers_array[k]
            for j in range(K):
                Q_j = cluster_centers_array[j]
                w_c[j] = self._return_color_weights(Q_k, Q_j)
                x_i_list.append(np.where(cluster_labels_array == j)[1])  # taking the column indeces
                y_i_list.append(np.where(cluster_labels_array == j)[0])  # taking the rows indeces
            # To be faster it estimates the denominator only once
            denominator = np.sum(np.multiply(bincount_array, w_c))
            # Estimate the center
            m_xk[k] = np.sum(w_c * np.sum(x_i_list[k])) / denominator
            m_yk[k] = np.sum(w_c * np.sum(y_i_list[k])) / denominator
            # Estimate the variance
            V_xk[k] = np.sum(w_c * np.sum((x_i_list[k] - m_xk[k])**2)) / denominator
            V_yk[k] = np.sum(w_c * np.sum((y_i_list[k] - m_yk[k])**2)) / denominator
        return m_xk, m_yk, V_xk, V_yk, w_c

    def _return_color_weights(self, C_i, C_j, sigma=16):
        """
        
        @param C_i: 
        @param C_j: 
        @param omega: parameter to adjust the effect of the color difference. 
        @return: 
        """
        numerator = (C_i[0] - C_j[0])**2 + (C_i[1] - C_j[1])**2 + (C_i[2] - C_j[2])**2
        numerator = np.sqrt(numerator)
        # numerator = np.linalg.norm(C_i-C_j)**2  # squared euclidean distance
        #denominator = 2*sigma*sigma  # normalisation
        return np.exp(-numerator/(2*sigma*sigma))

    def _return_saliency_and_contrast(self, labels_array, tot_bins, n_w, n_h, m_x, m_y, V_x, V_y, cluster_centers_array, bincount_array):
        N = n_w * n_h #image dimension
        label_matrix = np.reshape(labels_array, (n_h, n_w))
        #Variables for the saliency probability
        image_salient = np.zeros((n_h, n_w))
        g_list = list()
        #Variable for the global contrast
        image_contrast = np.zeros((n_h, n_w))
        h_w = np.zeros((tot_bins,tot_bins))
        for k in range(tot_bins):
            #Calculate the g vectors for the saliency probability
            g = np.array([np.sqrt(12 *V_x[k]) / n_w,
                          np.sqrt(12 * V_y[k]) / n_h,
                          (m_x[k]-(n_w/ 2))/n_w,
                          (m_y[k]-(n_h/ 2))/n_h])
            g_list.append(g)
            for j in range(tot_bins):
                #calculate the distances for the contrast
                Q_k = cluster_centers_array[k]
                Q_j = cluster_centers_array[j]
                h_w[k,:] = bincount_array[j] * self._return_color_weights(Q_k, Q_j)

        #for each pixel finds the probability of saliency
        for col in range(n_w):
            for row in range(n_h):
                bin_index = label_matrix[row,col]
                g = g_list[bin_index]
                image_salient[row,col] = (1 / ((2 * np.pi)**2 * np.sqrt(self.determinant_covariance))) * \
                                         np.exp(-np.dot(np.dot((g-self.mean_vector), self.covariance_matrix_inverse), g-self.mean_vector)/2)

                # Find the contrast image
                image_contrast[row, col] = np.sum(h_w[bin_index,:])

        return image_salient, image_contrast

    # Saliency probability vector
    def _return_p_vector(self, n_w, n_h, m_x, m_y, V_x, V_y):
        # Variables for the saliency probability
        tot_bins = np.amax(m_x.shape)
        p_vector = np.zeros(tot_bins)
        for k in range(tot_bins):
            # Calculate the g vectors for the saliency probability
            g = np.array([np.sqrt(12 *V_x[k]) / n_w,
                          np.sqrt(12 * V_y[k]) / n_h,
                          (m_x[k]-(n_w/ 2))/n_w,
                          (m_y[k]-(n_h/ 2))/n_h])
            # Allocare the result in the p vector
            p_vector[k] = (1 / ((2 * np.pi) ** 2 * np.sqrt(self.determinant_covariance))) * \
                            np.exp(-np.dot(np.dot((g - self.mean_vector), self.covariance_matrix_inverse),
                            g - self.mean_vector) / 2)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(p_vector)
        p_vector = p_vector - minVal
        p_vector = p_vector / (maxVal - minVal + 1e-3)
        return p_vector

    # Contrast vector
    def _return_r_vector(self, w_matrix, bincount_array):
        r_vector = np.sum(bincount_array * w_matrix, axis=1)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(r_vector)
        r_vector = r_vector - minVal
        r_vector = r_vector / (maxVal - minVal + 1e-3)
        return r_vector

    def _return_r_vector_mod(self, centers_array, bincount_array):
        distance_matrix = self._return_distance_matrix(centers_array)
        r_vector = np.sum(bincount_array * distance_matrix, axis=1)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(r_vector)
        r_vector = r_vector - minVal
        r_vector = r_vector / (maxVal - minVal + 1e-3)
        return r_vector

    # The final saliency vector
    def _return_s_vector(self, w_matrix, p_vector, r_vector):
        w_sum = np.sum(w_matrix, axis=1)
        s_vector = (w_sum * p_vector * r_vector) / w_sum
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(s_vector)
        s_vector = s_vector - minVal
        s_vector = s_vector / (maxVal - minVal + 1e-3)
        return s_vector

    def _return_w_matrix(self, centers_array):
        tot_bins = np.amax(centers_array.shape)
        w_matrix = np.zeros((tot_bins, tot_bins))
        for k in range(tot_bins):
            Q_k = centers_array[k]
            for j in range(tot_bins):
                Q_j = centers_array[j]
                w_matrix[k, j] = self._return_color_weights(Q_k, Q_j)
        return w_matrix

    def _return_distance_matrix(self, centers_array):
        tot_bins = np.amax(centers_array.shape)
        w_matrix = np.zeros((tot_bins, tot_bins))
        for k in range(tot_bins):
            Q_k = centers_array[k]
            for j in range(tot_bins):
                Q_j = centers_array[j]
                w_matrix[k, j] = np.sqrt((Q_k[0] - Q_j[0])**2 + (Q_k[1] - Q_j[1])**2 + (Q_k[2] - Q_j[2])**2)
        return w_matrix

    def _map_labels_to_vector(self, labels_array, vector, n_w, n_h):
        image = np.zeros((n_h,n_w))
        labels_matrix = labels_array.reshape((n_h,n_w))
        #for each pixel finds the probability of saliency
        for col in range(n_w):
            for row in range(n_h):
                bin_index = labels_matrix[row, col]
                image[row, col] = vector[bin_index]

        image = 255 * image
        #return cv2.convertScaleAbs(image)
        return np.uint8(image)

def main():
    my_map = FasaSaliencyMapping()
    image = cv2.imread("/home/massimiliano/Desktop/squirrel.png")
    image_lab, image_quantized, center, label, bincount_array = my_map._return_quantized_image(image)
    #print(center)
    #print(bincount_array)
    #print(label)
    m_x, m_y, V_x, V_y, w_c = my_map._return_centers_variances(center, label, bincount_array)
    #print(m_x)
    #print(m_y)
    #print(V_x)
    #print(V_y)
    n_h = image.shape[0]
    n_w = image.shape[1]
    #image_saliency, image_contrast = my_map._return_saliency_and_contrast(label, 8, n_w, n_h, m_x, m_y, V_x, V_y, center, bincount_array)

    w_matrix = my_map._return_w_matrix(center)
    print(w_matrix)
    print("")
    p_vector = my_map._return_p_vector(n_w, n_h, m_x, m_y, V_x, V_y)
    print(p_vector)
    print("")
    #r_vector = my_map._return_r_vector(w_matrix, bincount_array)
    r_vector = my_map._return_r_vector_mod(center, bincount_array)
    print(r_vector)
    print("")
    s_vector = my_map._return_s_vector(w_matrix, p_vector, r_vector)
    print(s_vector)
    print("")

    final_image = my_map._map_labels_to_vector(label, s_vector, n_w, n_h)

    cv2.rectangle(image, (int(m_x[0]),int(m_y[0])), (int(m_x[0]+5),int(m_y[0])+5), [0,0,255], 4)
    cv2.rectangle(image, (int(m_x[1]),int(m_y[1])), (int(m_x[1]+5),int(m_y[1])+5), [0,0,255], 4)
    cv2.rectangle(image, (int(m_x[2]),int(m_y[2])), (int(m_x[2]+5),int(m_y[2])+5), [0,0,255], 4)
    cv2.rectangle(image, (int(m_x[3]),int(m_y[3])), (int(m_x[3]+5),int(m_y[3])+5), [0,0,255], 4)
    cv2.rectangle(image, (int(m_x[4]),int(m_y[4])), (int(m_x[4]+5),int(m_y[4])+5), [0,0,255], 4)
    cv2.rectangle(image, (int(m_x[5]),int(m_y[5])), (int(m_x[5]+5),int(m_y[5])+5), [0,0,255], 4)
    cv2.rectangle(image, (int(m_x[6]),int(m_y[6])), (int(m_x[6]+5),int(m_y[6])+5), [0,0,255], 4)
    cv2.rectangle(image, (int(m_x[7]),int(m_y[7])), (int(m_x[7]+5),int(m_y[7])+5), [0,0,255], 4)
    cv2.imshow("image", image)
    cv2.imshow("saliency", final_image)
    #cv2.imshow("quantized", image_quantized)
    #cv2.imshow("lab", image_lab)
    while True:
        if cv2.waitKey(33) == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()