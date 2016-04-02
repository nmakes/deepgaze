import numpy
import cv2
import sys
from include.haar_cascade import haarCascade
from include.face_landmark_detection import faceLandmarkDetection

#Constant variables
TRIANGLE_POINTS = list(range(36,61))
DEBUG = False #If True enables the verbose mode


def main():

    #Defining the video capture object
    video_capture = cv2.VideoCapture(0)

    #Obtaining the CAM dimension
    cam_w = int(video_capture.get(3))
    cam_h = int(video_capture.get(4))

    #Declaring the two classifiers
    my_cascade = haarCascade("./etc/haarcascade_frontalface_alt.xml", "./etc/haarcascade_profileface.xml")
    my_detector = faceLandmarkDetection('./etc/shape_predictor_68_face_landmarks.dat')

    #Error counter definition
    no_face_counter = 0

    #Variables that identify the face
    # position in the main frame.
    face_x1 = 0
    face_y1 = 0
    face_x2 = 0
    face_y2 = 0
    face_w = 0
    face_h = 0

    #Variables that identify the ROI
    # position in the main frame.
    roi_x1 = 0
    roi_y1 = 0
    roi_x2 = cam_w
    roi_y2 = cam_h
    roi_w = cam_w
    roi_h = cam_h
    roi_resize_w = int(cam_w/10)
    roi_resize_h = int(cam_h/10)

    while(True):

        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame[roi_y1:roi_y2, roi_x1:roi_x2], cv2.COLOR_BGR2GRAY)

        #Looking for faces with cascade
        my_cascade.findFace(gray, True, True)

        #Accumulate error values in a counter
        if(my_cascade.face_type == 0): 
            no_face_counter += 1

        #If any face is found for a certain
        # number of cycles, then the ROI is reset
        if(no_face_counter == 50):
            no_face_counter = 0
            roi_x1 = 0
            roi_y1 = 0
            roi_x2 = cam_w
            roi_y2 = cam_h
            roi_w = cam_w
            roi_h = cam_h

        #Checking wich kind of face it is returned
        if(my_cascade.face_type > 0):

            #Updating the face position
            face_x1 = my_cascade.face_x + roi_x1
            face_y1 = my_cascade.face_y + roi_y1
            face_x2 = my_cascade.face_x + my_cascade.face_w + roi_x1
            face_y2 = my_cascade.face_y + my_cascade.face_h + roi_y1
            face_w = my_cascade.face_w
            face_h = my_cascade.face_h

            #Updating the ROI position       
            roi_x1 = face_x1 - roi_resize_w
            if (roi_x1 < 0): roi_x1 = 0
            roi_y1 = face_y1 - roi_resize_h
            if(roi_y1 < 0): roi_y1 = 0
            roi_w = face_w + roi_resize_w + roi_resize_w
            if(roi_w > cam_w): roi_w = cam_w
            roi_h = face_h + roi_resize_h + roi_resize_h
            if(roi_h > cam_h): roi_h = cam_h    
            roi_x2 = face_x2 + roi_resize_w
            if (roi_x2 > cam_w): roi_x2 = cam_w
            roi_y2 = face_y2 + roi_resize_h
            if(roi_y2 > cam_h): roi_y2 = cam_h

            #Debugging printing utilities
            if(DEBUG == True):
                print("FACE: ", face_x1, face_y1, face_x2, face_y2, face_w, face_h)
                print("ROI: ", roi_x1, roi_y1, roi_x2, roi_y2, roi_w, roi_h)

            #Drawing a green rectangle
            # around the detected face.
            cv2.rectangle(frame, 
                         (face_x1, face_y1), 
                         (face_x2, face_y2), 
                         (0, 255, 0),
                          2)

            #In case of a frontal face it
            # is called the landamark detector
            if(my_cascade.face_type == 1):
                matrixLandmarks = my_detector.returnLandmarks(frame, face_x1, face_y1, face_x2, face_y2)
                for row in TRIANGLE_POINTS:
                     cv2.circle(frame,( matrixLandmarks[row].item((0,0)), matrixLandmarks[row].item((0,1)) ), 2, (0,0,255), -1)

        #Drawing a yellow rectangle
        # around the ROI.
        cv2.rectangle(frame, 
                    (roi_x1, roi_y1), 
                    (roi_x2, roi_y2), 
                    (0, 255, 255),
                     2)

        #Showing the frame and waiting
        # for the exit command
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
   
    #Release the camera
    video_capture.release()




if __name__ == "__main__":
    main()
