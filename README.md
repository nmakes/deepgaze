Deepgaze 
----------

Gaze direction estimation using convolutional neural network.

Work in progress. The code provided at the moment do not still implement gaze recognition. In the next few weeks I count to provide the first official release. In `main.py` you can find how to estimate the head pose using the opencv function solvePnP. The next step will be to grab the eyes rectangle and run a convolutional neural network to estimate the gaze direction, then integrate the pose with the reference frame of the head.

Dependencies
------------

To run the code in main.py you have to install `opencv` and `numpy`.

Install
--------

1. clone the repository
2. open a terminal inside the main folder of the repository
3. to run the pose estimation example: python main.py
