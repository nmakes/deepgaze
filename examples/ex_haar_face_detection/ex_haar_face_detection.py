#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola, 2017 Luca Surace
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#Using the OpenCV haar cascade classifiers to find a face in image.

from deepgaze.face_detection import HaarFaceDetector
import cv2

# Reading the image in black/withe
image = cv2.imread("./group.jpg",0)

hfd = HaarFaceDetector("../../etc/xml/haarcascade_frontalface_alt.xml", "../../etc/xml/haarcascade_profileface.xml")
allTheFaces = hfd.returnMultipleFacesPosition(image, runFrontal=True, runFrontalRotated=True, 
                    runLeft=True, runRight=True, 
                    frontalScaleFactor=1.2, rotatedFrontalScaleFactor=1.2, 
                    leftScaleFactor=1.15, rightScaleFactor=1.15, 
                    minSizeX=64, minSizeY=64, 
                    rotationAngleCCW=30, rotationAngleCW=-30)

# Iterating all the faces 
for element in allTheFaces:
    face_x1 = int(element[0])
    face_y1 = int(element[1])
    face_x2 = int(face_x1+element[2])
    face_y2 = int(face_y1+element[3])
    cv2.rectangle(image, (face_x1, face_y1), (face_x2, face_y2), [255, 0, 0])

# Drawing a rectangle around the face
cv2.rectangle(image, (face_x1, face_y1), (face_x2, face_y2), [0,0,255])

# Showing the face and waiting for a key to exit
cv2.imshow("Face detected", image)
cv2.waitKey(0)

