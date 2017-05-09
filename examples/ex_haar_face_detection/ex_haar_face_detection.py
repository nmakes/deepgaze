#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#Using the OpenCV haar cascade classifiers to find a face in image.


from deepgaze.haar_cascade import haarCascade
import cv2

#Declaring the face detector object and loading the XML config file
my_cascade = haarCascade("../../etc/xml/haarcascade_frontalface_alt.xml", "../../etc/xml/haarcascade_profileface.xml")

#Reading the image in black/withe
image = cv2.imread("./bellucci.jpg",0)

#Calling the findFace method
my_cascade.findFace(image, runFrontal=True, runFrontalRotated=True, 
                    runLeft=True, runRight=True, 
                    frontalScaleFactor=1.2, rotatedFrontalScaleFactor=1.2, 
                    leftScaleFactor=1.15, rightScaleFactor=1.15, 
                    minSizeX=64, minSizeY=64, 
                    rotationAngleCCW=30, rotationAngleCW=-30)

#The coords of the face are saved in the class object
face_x1 = my_cascade.face_x
face_y1 = my_cascade.face_y
face_x2 = my_cascade.face_x + my_cascade.face_w
face_y2 = my_cascade.face_y + my_cascade.face_h
face_w = my_cascade.face_w 
face_h = my_cascade.face_h
 

# Print this when no face is detected
if(my_cascade.face_type == 0): 
    print("No face detected!")

#Drawing a rectangle around the face
cv2.rectangle(image, (face_x1, face_y1), (face_x2, face_y2), [0,0,255])

#showing the face and waiting for a key to exit
cv2.imshow("Face detected", image)
cv2.waitKey(0)

