#!/usr/bin/env python

## @package bayse_filter.py
#
# Massimiliano Patacchiola, Plymouth University 2016
#
# Discrete Bayes filter (DBF) implementation. It permits estimating
# the value of a quantity X given the observation Z.
# If we have some noisy measurement Z of a discrete quantity X, we  
# can use the DBF  in a recursive estimationto find the most 
# probable value of X (belief). 


import numpy as np
import sys


class DiscreteBayesFilter:

    def __init__(self, states_number):
        if(states_number<=0):
            raise ValueError('BayesFilter: the number of states must be greater than zero.')
        else: 
            self._states_number = states_number
            #Declaring the prior distribution
            self._prior = np.zeros(states_number, dtype=np.float32)
            self._prior.fill(1.0/states_number) #uniform
            #Declaring the poterior distribution
            #self._posterior = np.empty(states_number, dtype=np.float32)
            #self._posterior.fill(1.0/states_number) #uniform
            #Declaring the conditional probability table
            self._cpt = np.zeros((states_number, states_number), dtype=np.float32)
            self._cpt.fill(1.0/states_number) #uniform
            #Declaring the evidence scalar
            #self._evidence = -1 #negative means undefined

    ##
    # Return the posterior probability given a prior and an evidence.
    # @param prior is a 1 dimensional array, it represents the distribution of X at t_0
    # @param cpt is a matrix, it is the conditional probability table of Z|X
    def initialise(self, prior, cpt):
        if(prior.shape[0]!=self._states_number):
            raise ValueError('DiscreteBayesFilter: the shape of the prior is different from the total number of states.')
        elif(cpt.shape[0]!=self._states_number or cpt.shape[1]!=self._states_number):
            raise ValueError('DiscreteBayesFilter: the shape of the cpt is different from the total number of states.')
        else: 
            self._prior = prior.copy()
            self._cpt = cpt.copy()

    ##
    # After the initialisation it is possible to predict the current value
    # of the quanity X. It is necessary to pass the cpt and the prior in the
    # initialisation phase. After the prediction this function update the 
    # internal prior with the posterior just computed. Basically we are 
    # applying the Bayes theorem to estimate the posterior.
    # @param evidence is a scalar, it represents the observed state of Z
    # @return it returns the posterior distribution of X given the evidence
    def predict_and_update(self, evidence):
        #Getting P(Z) and Likelihood
        p_z = np.sum(self._cpt[:,evidence]) #scalar P(Z)
        likelihood = self._cpt[:,evidence].copy() #vector P(Z|X)
        #likelihood = np.transpose(likelihood)
        #Getting the posterior distribution
        posterior = np.multiply(self._prior, likelihood)
        posterior /= p_z #vector P(X|Z)
        #Normalise to sum up to 1
        normalisation = np.sum(posterior)
        posterior /= normalisation
        #Update the posterio with the new value
        self._prior = posterior
        return posterior.copy()






    
