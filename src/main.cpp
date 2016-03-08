/**
 * @file objectDetection.cpp
 * http://www.instructables.com/id/Face-detection-and-tracking-with-Arduino-and-OpenC/
 * Based on code writtenby A. Huaman ( based in the classic facedetect.cpp in samples/c )
 * @brief A simplified version of facedetect.cpp, show how to load a cascade classifier and how to find objects (Face + eyes) in a video stream
 * Mdified to intercept X,Y of center of face 
 */
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


#include<signal.h>
#include <iostream>
#include <cstdio>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
String face_cascade_name = "../etc/xml/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "../etc/xml/haarcascade_eye.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture Face detection";


int main( int argc, const char** argv )
{
 CvCapture* capture;
 cv::Mat frame;

 //-- 1. Load the cascades
 if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
 if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
 
 //-- 2. Read the video stream
 capture = cvCaptureFromCAM(-1); //the -1 search for the video streaming when the ID is unknown

 if(!capture) return 0;

 while( true )
 {
  frame = cvQueryFrame( capture );
  //-- 3. Apply the classifier to the frame
  if( !frame.empty() )
  { 
   detectAndDisplay( frame ); 
  } else { 
   printf(" --(!) No captured frame -- Break!"); 
   break; 
  }
      
  int c = waitKey(10);
  if( (char)c == 'c' ) { 
    break; 
  } 
 }

 return 0;
}

/**
 * @function detectAndDisplay
 */
void detectAndDisplay( Mat frame )
{
 
 std::vector<Rect> faces;
 std::vector<Rect> eyes;
 Mat frame_gray;
 Mat faceROI;
 Mat eyeROI;
 cvtColor( frame, frame_gray, CV_BGR2GRAY );
 equalizeHist( frame_gray, frame_gray );
 //-- Detect faces: detectMultiScale()
 //image		Matrix of the type CV_8U containing an image where objects are detected.
 //objects		Vector of rectangles where each rectangle contains the detected object, the rectangles may be partially outside the original image.
 //scaleFactor		Parameter specifying how much the image size is reduced at each image scale.
 //minNeighbors		Parameter specifying how many neighbors each candidate rectangle should have to retain it.
 //flags		Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. It is not used for a new cascade.
 //minSize		Minimum possible object size. Objects smaller than that are ignored.
 //maxSize		Maximum possible object size. Objects larger than that are ignored.
 face_cascade.detectMultiScale( frame, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30), Size(300, 300) );

 if(faces.size() == 0){
  imshow( window_name, frame );
  return;
 }

 int max_area_index = -1;
 double max_area_value = 0;
 for( int i = 0; i < faces.size(); i++ ){
  if(faces[i].area() > max_area_value) max_area_index = i;
 }

 
 //for( int i = 0; i < faces.size(); i++ )
  //{
      //Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
      //ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 2, 8, 0 );    
      cv::rectangle(frame, faces[max_area_index], Scalar(255, 0, 0), 2); //Drawing a rectangle in frame, size of faces[], Scalar color, tickness.

      
    faceROI = frame_gray(faces[max_area_index]);
    //Searching only in the upper part of the matrix
    faceROI = faceROI.adjustROI(0, (int)-faceROI.rows/2, 0, 0);

   //-- Detect eyes
   eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(20, 20), Size(50, 50));
   if(eyes.size() > 2) eyes.resize(2);

   for( int j = 0; j < eyes.size(); j++ )
   {
      Point center( faces[max_area_index].x + eyes[j].x + eyes[j].width*0.5, faces[max_area_index].y + eyes[j].y + eyes[j].height*0.5 );
      int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
      circle( frame, center, radius, Scalar( 255, 255, 0 ), 4, 8, 0 );
      //ellipse( frame, center, Size( eyes[j].width*0.5, eyes[j].height*0.5), 0, 0, 360, Scalar( 255, 255, 255 ), 2, 8, 0 );
      //cv::rectangle(frame, eyes[j], Scalar(255, 255, 255), 2);
      //Point center( eyes[i].x + eyes[j].width*0.5, eyes[j].y + eyes[j].height*0.5 );
      //ellipse( frame, center, Size( eyes[j].width*0.5, eyes[j].height*0.5), 0, 0, 360, Scalar( 255, 255, 255 ), 2, 8, 0 );
      //rectangle(frame, center, Size( eyes[i].width*0.5, eyes[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 2, 8, 0);
      //  cout << "X:" << faces[i].x  <<  "  y:" << faces[i].y  << endl;
      eyeROI = frame_gray( eyes[j] );
    } 
   //}

   //-- Show what you got
   namedWindow( window_name, WINDOW_AUTOSIZE );// Create a window for display.
   imshow( window_name, frame );
   //imshow( "Face Detected" , faceROI );
}


/**
 * @function detectAndDisplay
 */
cv::Mat detectEye(cv::Mat frame )
{



}















