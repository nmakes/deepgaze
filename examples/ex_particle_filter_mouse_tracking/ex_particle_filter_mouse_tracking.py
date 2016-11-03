#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#Example of particle filtering estimation of a moving point.
#In this example the mouse pointer represent the moving object.
#The filter will generate some particles which are candidates
#of the position to estimate. A dirty measurement of the mouse
#position is then given as input in the updating phase. The
#filter is robust to wrong measurement and can keep tracking
#the mouse pointer. To iterate one time move the mouse on the
#window and click with the left button. If you keep clicking
#you will see the blue point (estimation) and the red points
#(particles) approaching the green dot (mouse pointer).

import cv2
import numpy as np
from deepgaze.motion_tracking import ParticleFilter

#Dimensions of the window
height = 512
width = 800
#Filter parameters
tot_particles = 100
#Standard deviation which represent how to spread the particles
#in the prediction phase.
std = 25 
#Init the filter
my_particle = ParticleFilter(height, width, tot_particles)
#Black image, window, bind the function
img = np.zeros((height,width,3), np.uint8)
cv2.namedWindow('image')

#Mouse callback function
def draw_circle(event,x,y,flags,param):
    #if event == cv2.EVENT_MOUSEMOVE:
    if event == cv2.EVENT_LBUTTONDOWN:

        #Predict the position of the pointer
        my_particle.predict(x_velocity=0, y_velocity=0, std=std)

        #Estimate the next position using the internal model
        x_estimated, y_estimated, _, _ = my_particle.estimate() 
 
        #Update the position of the particles based on the measurement.
        #Adding some noise to the measurement.
        noise_coefficient = np.random.uniform(low=0.0, high=10.0)
        x_measured = x + np.random.randn() * noise_coefficient
        y_measured = y + np.random.randn() * noise_coefficient
        my_particle.update(x_measured, y_measured)

        #Drawing the circles for the mouse position the
        #estimation and the particles.
        for i in range(0, tot_particles):
            x_particle, y_particle = my_particle.returnParticlesCoordinates(i)
            cv2.circle(img,(x_particle, y_particle),2,(0,0,255),-1) #RED: Particles
        cv2.circle(img,(x, y),2,(0,255,0),-1) #GREEN: Mouse position
        cv2.circle(img,(x_estimated, y_estimated),2,(255,0,0),-1) #BLUE: Filter estimation

        #Print general information
        print("Total Particles: " + str(tot_particles))
        print("Effective N: " + str(my_particle.returnParticlesContribution()))
        print("Measurement Noise: " + str(noise_coefficient) + "/10")
        print("x=" + str(x) + "; y=" + str(y) + " | " + 
              "x_measured=" + str(int(x_measured)) + "; y_measured=" + str(int(y_measured))  + " | " +
              "x_estimated=" + str(int(x_estimated)) + "; y_estimated=" + str(int(y_estimated)))
        #print(my_particle.weights) #uncomment to print the weights
        #print(my_particle.particles) #uncomment to print the particle position
        print("")

        #if(my_particle.returnParticlesContribution() < 8):
        my_particle.resample()


cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    #Press ESC to stop the simulation
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()
