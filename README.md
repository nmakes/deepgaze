What is deepgaze?
----------
Deepgaze is a library which uses Convolutional Neural Networks (CNNs) to estimate the Focus of Attention (FOA) of users. The FOA can be approximately estimated finding the head orientation. This is particularly useful when the eyes are covered, or when the user is too far from the camera to grab the eye region with a good resolution. When the eye region is visible it is possible to estimate the gaze direction, which is much more informative and can give a good indication of the FOA.

**Update 15/08/2016**:
Work in progress. The code provided at the moment does not still implement gaze detection. There is an example for using the OpenCV solvePnP method to estimate the head orientation [[code]](./examples/ex_pnp_head_pose_estimation.py). The code for head pose estimation with CNN will be available in September, and the code for gaze estimation before the end of the year.


What is a Convolutional Neural Network?
------------------------------
A convolutional neural network (CNN, or ConvNet) is a type of feed-forward artificial neural network in which the connectivity pattern between its neurons is inspired by the organization of the animal visual cortex, whose individual neurons are arranged in such a way that they respond to overlapping regions tiling the visual field. Convolutional networks were inspired by biological processes and are variations of multilayer perceptrons designed to use minimal amounts of preprocessing. They have wide applications in image and video recognition, recommender systems and natural language processing [[wiki]](https://en.wikipedia.org/wiki/Convolutional_neural_network)


Prerequisites
------------
To use the libray you have to install:

- Numpy [[link]](http://www.numpy.org/)

```shell
sudo pip install numpy
```

- OpenCV [[link]](http://opencv.org/)

```shell
sudo apt-get install libopencv-dev python-opencv
```

- Tensorflow [[link]](https://www.tensorflow.org/)

```shell
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl
sudo pip install --upgrade $TF_BINARY_URL
```

Some examples may require additional libraries:

- dlib [[link]](http://dlib.net/)

Installation
--------

Download the repository from [[here]](https://github.com/mpatacchiola/deepgaze/archive/master.zip) or clone it using git:

```shell
git clone https://github.com/mpatacchiola/deepgaze.git
```

To install the package you have to run the setup.py script (it may require root privileges):

```shell
python setup.py install
```

Done! Now give a look to the examples folder.

Examples
--------

- Head Pose Estimation using the Perspective-n-Point algorithm in OpenCV [[code]](./examples/ex_pnp_head_pose_estimation_webcam.py) [[video]](https://www.youtube.com/watch?v=OSnI18XmAg4)









