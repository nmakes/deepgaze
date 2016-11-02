#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#Thank to rlabbe and his fantastic repository for Bayesian Filter:
#https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

import numpy as np
from numpy.random import uniform
import cv2
import sys

class ParticleFilter:
    """Particle filter motion tracking.

    This class estimates the position of a single point
    in a image. It can be used to predict the position of a
    landmark for example when tracking some face features.
    """

    def __init__(self, width, height, N):
        """Init the particle filter.

        @param width the width of the frame
        @param height the height of the frame
        @param N the number of particles
        """
        if(N <= 0): raise ValueError('[DEEPGAZE] motion_tracking.py: the ParticleFilter class does not accept a value of N which is <= 0')
        self.particles = np.empty((N, 2))
        self.particles[:, 0] = uniform(0, width, size=N) #init the X coord
        self.particles[:, 1] = uniform(0, height, size=N) #init the Y coord
        #Init the weiths vector as a uniform distribution
        #at the begining each particle has the same probability
        #to represent the point we are following
        self.weights = np.empty((N, 1))
        self.weights.fill(1.0/N) #normalised values

    def predict(self, x_speed, y_speed, std ):
        """Predict the position of the point in the next frame.
 
        The position of the point at the next time step is predicted using the 
        estimated speed along X and Y axis and adding Gaussian noise sampled 
        from a distribution with MEAN=0.0 and STD=std. It is a linear model.
        @param x_speed the speed of the object along the X axis in terms of pixels/frame
        @param y_speed the speed of the object along the Y axis in terms of pixels/frame
        @param std the standard deviation of the gaussian distribution used to add noise
        """
        #To predict the position of the point at the next step we take the
        #previous position and we add the estimated speed and Gaussian noise
        self.particles[:, 0] += x_speed + (np.random.randn(len(self.particles)) * std) #predict the X coord
        self.particles[:, 1] += y_speed + (np.random.randn(len(self.particles)) * std) #predict the Y coord

    def update(self, x, y ):
        """Update the weights associated which each particle based on the (x,y) coords measured.
 
        The position of the point at the next time step is predicted using the 
        estimated speed along X and Y axis and adding Gaussian noise sampled 
        from a distribution with MEAN=0.0 and STD=std. It is a linear model.
        @param x the position of the point in the X axis
        @param y the position of the point in the Y axis
        @param 
        """
        #TODO complete this function
        distance = np.linalg.norm(self.particles - [x,y], axis=1)
        self.weights = 

    def estimate(self):
        """Estimate the position of the point given the particle weights.
 
        This function get the mean and variance associated with the point to estimate.
        @return get the x_mean, y_mean and the x_var, y_var 
        """
        #Using the weighted average of the particles
        #gives an estimation of the position of the point
        mean = np.average(self.particles, weights=self.weights, axis=0)
        var  = np.average((self.particles - mean)**2, weights=self.weights, axis=0)
        x_mean = int(mean[0])
        y_mean = int(mean[1])
        x_var = int(var[0])
        y_var = int(var[1])
        return x_mean, y_mean, x_var, y_var

    def returnParticlesContribution(self):
        """This function gives an estimation of the number of particles which are
        contributing to the probability distribution (also called the effective N). 
 
        This function get the effective N value which is a good estimation for
        understanding when it is necessary to call a resampling step. When the particle
        are collapsing in one point only some of them are giving a contribution to 
        the point estimation. If the value is less than N/2 then a resampling step
        should be called. A smaller value means a larger variance for the weights, 
        hence more degeneracy
        @return get the effective N value. 
        """
        return 1.0 / np.sum(np.square(self.weights))

