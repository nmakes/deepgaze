
In this example it is possible to train the weights of Convolutional Neural Networks (CNNs) used in the article:

Li, H., Lin, Z., Shen, X., Brandt, J., & Hua, G. (2015). *A convolutional neural network cascade for face detection.* In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 5325-5334).

The article present a cascade of CNNs where 3 networks operate a **detection** and other 3 operate a **calibration** of the boundix boxes. Here it is possible to train the 6 networks and use the weights in Deepgaze.

Usage
------

1. Download a dataset containing human faces which gives the bounding boxes. An example could be the AFLW dataset.

2. Images generation for detection. For the **detection dataset** you need positive and negative images containing human faces and random background patches. You can collect positive images (faces) using the script called **extract_faces.py**. Negative images (non-faces) can be easilly collected from background.

3. Images generation for calibration. The calibration CNNs requires images of faces shifted in a particular way.

4. Dataset generation. This step is the same for the creation of the dataset for both detection and generation. Here it is necessary to have positive and negative images saved inside a folder. The images will be preprocessed and converted to a pickle file which is used during the training procedure. For this step you can use the two scripts: **preprocess_positive.py** and **preprocess_negative.py**. The script must be called with the following arguments: *-s* (imgage size) which must be one of: 12, 24 or 48 in accordance with the network you want to train. The second argument to pass is *-i* (dataset path) which should point to the foder containing the images. The scripts will look for any **JPG image** present in any folder and sub-folder of the specified root directory. Calling the command: `python preprocess_positive.py -s=12 -i="./faces"` the script will look to any JPG images in the *./faces* folder and will resize them to be 12x12.

5. Train CNNs for detection. Run the scripts: **12net_detection_training.py**, **24net_detection_training.py**, **48net_detection_training.py**. Those scripts will save the network weights which will be used later. To run the scripts you have to modify the variables `pickle_file_positive` and `pickle_file_negative` with a path to the two pickle files created in the previous step. You may also change other variables like `tot_epochs` (total number of epochs), `batch_size` (number of images per batch), `tf_initializer` (weights initializer, default is Glorot), optimizer type, etc.

6. Train CNNs for calibration. Run the scripts: **12net_calibration_training.py**, **24net_calibration_training.py**, **48net_calibration_training.py**. Similarly to the previous step to run the scripts you have to modify the variables `pickle_file_positive` and `pickle_file_negative` with a path to the two pickle files created before. You may also change other variables like `tot_epochs` (total number of epochs), `batch_size` (number of images per batch), `tf_initializer` (weights initializer, default is Glorot), optimizer type, etc.

7. TODO: load the weights and use the networks.





