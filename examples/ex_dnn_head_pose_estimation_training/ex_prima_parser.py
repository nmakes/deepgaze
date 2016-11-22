#!/usr/bin/env python

##
# Massimiliano Patacchiola, Plymouth University 2016
# massimiliano.patacchiola@plymouth.ac.uk
#
# Python example for the manipulation of the Prima Head Pose Image Database:
# http://www-prima.inrialpes.fr/perso/Gourier/Faces/HPDatabase.html
#
# It contains three functions that allow creating a CSV file and to crop/resize
# the faces. It can generate 15 pickle files for the leave-one-out (Jack Knife)
# cross-validation test on unknown subjects. 
# To use the file you have to insert the right paths in the main function.
#
# Requirements: 
# OpenCV (sudo apt-get install libopencv-dev python-opencv) 
# Numpy (sudo pip install numpy)
# Six (sudo pip install six)

import cv2
import os.path
import numpy as np
import csv
import glob
import re
from six.moves import cPickle as pickle

##
# Given the input directory containing the image folders (Person01, Person02, Person03, etc)
# it generates a CSV (comma separated value) files containing the image address and the 
# pan-tilt values. The images are cropped and the face is saved in the output folder.
# @param input_path the folder containing the database folders
# @param the output directory to use for saving the CSV files
# @param img_size crop the face and resize it to this size
# @param colour when True it save the cropped image in colour otherwise in grayscale
# @normalisation the angle values are normalised between 0 and 1
def create_csv(input_path, output_path, img_size=64, colour=True, normalisation=False):

    #Image counter
    counter = 0
    roll = 0.0

    #Create the output folder if does not find it
    if not os.path.exists(output_path): os.makedirs(output_path)

    #Write the header
    fd = open(output_path + '/prima_label.csv','w')
    fd.write("path, id, serie, tilt, pan" + "\n")
    fd.close()

    #Iterate through all the folder specified in the input path
    for folder in os.walk(input_path + "/"):
        for image_path in glob.glob(str(folder[0]) + "/*.jpg"):

            #Check if there are folders which not contain the 
            #substring "Person". If there are then skip them.
            splitted = str(folder[0]).split('/')
            folder_name = splitted[len(splitted)-1]
            if(("Person" in folder_name) == False): break;

            #Split the image name
            splitted = image_path.split('/')
            image_name = splitted[len(splitted)-1]
            file_name = image_name.split(".")[0]
            print("")
            print(file_name)
            #Regular expression to split the image string
            matchObj = re.match( r'(person)(?P<id>[0-9][0-9])(?P<serie>[12])(?P<number>[0-9][0-9])(?P<tilt>[+-][0-9]?[0-9])(?P<pan>[+-][0-9]?[0-9])', file_name, re.M|re.I)

            print("COUNTER: " + str(counter))
            print(image_path)
            print(matchObj.group(0))
            print("ID: " + matchObj.group("id"))
            print("SERIE: " + matchObj.group("serie"))
            print("NUMBER: " + matchObj.group("number"))
            print("TILT: " + matchObj.group("tilt"))
            print("PAN: " + matchObj.group("pan"))

            person_id = matchObj.group("id")
            person_serie = matchObj.group("serie")
            tilt = int(matchObj.group("tilt"))
            pan = int(matchObj.group("pan"))

            #Take the image information from the associated txt file        
            f=open(folder[0] +"/" + file_name + ".txt")
            lines=f.readlines()
            face_centre_x = int(lines[3])
            face_centre_y = int(lines[4])
            face_w = int(lines[5])
            face_h = int(lines[6])
            f.close

            #Take the largest dimension as size for the face box
            if(face_w > face_h):
                face_h = face_w
            if(face_h > face_w):
               face_w = face_h
            face_x = face_centre_x - (face_w/2)
            face_y = face_centre_y - (face_h/2)

            #Correction for aberrations
            if(face_x < 0):
               face_x = 0
            if(face_y < 0):
               face_y = 0

            #print("C_X: " + str(face_centre_x))
            #print("C_Y: " + str(face_centre_y))
            #print("W: " + str(face_w))
            #print("H: " + str(face_h))
            #print("X: " + str(face_x))
            #print("Y: " + str(face_y))      

            #Load the image (colour or grayscale)
            if(colour==True): image = cv2.imread(image_path) #load in colour
            else: image = cv2.imread(image_path, 0) #load in grayscale
            #Crop the face from the image
            image_cropped = np.copy(image[face_y:face_y+face_h, face_x:face_x+face_w])
            #Rescale the image to the predifined size
            image_rescaled = cv2.resize(image_cropped, (img_size,img_size), interpolation = cv2.INTER_AREA)
            #Create the output folder if does not find it
            if not os.path.exists(output_path + "/" + str(person_id)): os.makedirs(output_path + "/" + str(person_id))
            #Save the image
            output_dir = output_path + "/" + str(person_id) + "/" + str(int(person_id)) + "_" + str(tilt) + "_" + str(pan) + "_" + str(counter) + ".jpg"
            cv2.imwrite(output_dir, image_rescaled)

            #Write the CSV file for pan
            if(pan == -90): label_pan = -90
            elif(pan == -75): label_pan = -75
            elif(pan == -60): label_pan = -60
            elif(pan == -45): label_pan = -45
            elif(pan == -30): label_pan = -30
            elif(pan == -15): label_pan = -15
            elif(pan ==   0): label_pan = 0
            elif(pan == +15): label_pan = 15
            elif(pan == +30): label_pan = 30
            elif(pan == +45): label_pan = 45
            elif(pan == +60): label_pan = 60
            elif(pan == +75): label_pan = 75
            elif(pan == +90): label_pan = 90
            else: raise ValueError('ERROR: The pan is out of range ... ' + str(pan))

            #Write the CSV file for tilt
            if(tilt == -90): label_tilt = -90
            elif(tilt == -60): label_tilt = -60
            elif(tilt == -30): label_tilt = -30
            elif(tilt == -15): label_tilt = -15
            elif(tilt ==   0): label_tilt = 0
            elif(tilt == +15): label_tilt = 15
            elif(tilt == +30): label_tilt = 30
            elif(tilt == +60): label_tilt = 60
            elif(tilt == +90): label_tilt = 90
            else: raise ValueError('ERROR: The tilt is out of range ... ' + str(tilt))

            #pan-tilt Normalisation
            #Normalise between 0 and 1
            if(normalisation == True):
                label_pan += 90.0
                label_pan /= 180.0
                label_tilt += 90.0
                label_tilt /= 180.0

            #Write the CSV file
            fd = open(output_path + '/prima_label.csv','a')
            fd.write(output_dir + "," + str(int(person_id)) + "," + str(int(person_serie)) + "," + str(label_tilt) + "," + str(label_pan) + "\n")
            fd.close()

            counter += 1

##
# Generate a pickle file containing Numpy arrays ready to use for
# the Leave-One-Out (loo) coross-validation test. There are 15 pickle files.
# In each pickle file there is a test matrix containing the images of a 
# single subject and a training matrix containing the images of all 
# the other subjects.
# @param csv_path the path to the CSV file generated with create_csv function
# @param output_path the path where saving the 15 pickle files
# @param shuffle if True it randomises the position of the images in the training dataset
def create_loo_pickle(csv_path, output_path, shuffle=False):

    #Saving the TEST file names in a list
    image_list = list()
    with open(csv_path, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        first_line = 0 #To jump the header line
        for row in reader:
            if(first_line != 0): image_list.append(row[0])
            first_line = 1

    #Loading the labels
    person_id_vector = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=(1), dtype=np.float32)
    person_serie_vector = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=(2), dtype=np.float32)
    tilt_vector = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=(3), dtype=np.float32)
    pan_vector = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=(4), dtype=np.float32)

    #Printing shape
    print("Tot Images: " + str(len(image_list)))
    print("Person ID: " + str(person_id_vector.shape))
    print("Person Serie: " + str(person_serie_vector.shape))
    print("Tilt: " + str(tilt_vector.shape))
    print("Pan: " + str(pan_vector.shape))

    for i in range(1,15):

        #local variable clered at ech cycle
        training_list = list()
        training_tilt_list = list()
        training_pan_list = list()

        test_list = list()
        test_tilt_list = list()
        test_pan_list = list()
        row_counter = 0

        for person_id in person_id_vector:
            #Check if the image exists
            if os.path.isfile(image_list[row_counter]):
                image = cv2.imread(str(image_list[row_counter])) #colour
                img_h, img_w, img_d = image.shape
            else:
                print("The image do not exist: " + image_list[row_counter])
                raise ValueError('Error: the image file do not exist.')
 
            #Separate test and training sets          
            if(int(person_id) == i): 
                 test_list.append(image)
                 test_tilt_list.append(tilt_vector[row_counter])
                 test_pan_list.append(pan_vector[row_counter])
            else:
                 training_list.append(image)         
                 training_tilt_list.append(tilt_vector[row_counter]) 
                 training_pan_list.append(pan_vector[row_counter]) 
            row_counter += 1


        #Create arrays
        training_array = np.asarray(training_list)
        training_tilt_array = np.asarray(training_tilt_list) 
        training_pan_array = np.asarray(training_pan_list)
    
        test_array = np.asarray(test_list)
        test_tilt_array = np.asarray(test_tilt_list) 
        test_pan_array = np.asarray(test_pan_list) 

        training_array = np.reshape(training_array, (-1, img_h*img_w*img_d)) 
        training_tilt_array = np.reshape(training_tilt_array, (-1, 1)) 
        training_pan_array = np.reshape(training_pan_array, (-1, 1))
     
        test_array = np.reshape(test_array, (-1, img_h*img_w*img_d)) 
        test_tilt_array = np.reshape(test_tilt_array, (-1, 1)) 
        test_pan_array = np.reshape(test_pan_array, (-1, 1)) 

        print("Training dataset: ", training_array.shape)
        print("Training Tilt label: ", training_tilt_array.shape)
        print("Training Pan label: ", training_tilt_array.shape)
        print("Test dataset: ", test_array.shape)
        print("Test Tilt label: ", test_tilt_array.shape)
        print("Test Pan label: ", test_pan_array.shape)

        #Shuffle the Training dataset
        if(shuffle == True):
            #Temporary append the label to the dataset to shuffle the data
            #data = np.append(training_array, training_tilt_array, axis=1)
            data = np.concatenate((training_array, training_tilt_array, training_pan_array), axis=1)
            print("DATA shape: " + str(data.shape))
            #Shuffle the row to randomize the data
            np.random.shuffle(data)

            #Separating the label from the dataset
            leght = img_h*img_w*img_d
            training_array = data[:,0:leght]
            training_tilt_array = data[:,leght:leght+1]
            training_pan_array = data[:,leght+1:leght+2]

        #saving the dataset in a pickle file
        pickle_file = output_path + "/prima_p" + str(i) + "_out.pickle"
        print("Saving the dataset in: " + pickle_file)
        print("... ")
        try:
             print("Opening the file...")
             f = open(pickle_file, 'wb')
             save = {
               'training_dataset': training_array,
               'training_tilt_label': training_tilt_array,
               'training_pan_label': training_pan_array,    
               'test_dataset': test_array,
               'test_tilt_label': test_tilt_array,
               'test_pan_label': test_pan_array    
                   }

             print("Training dataset: ", training_array.shape)
             print("Training Tilt label: ", training_tilt_array.shape)
             print("Training Pan label: ", training_pan_array.shape)
             print("Test dataset: ", test_array.shape)
             print("Test Tilt label: ", test_tilt_array.shape)
             print("Test Pan label: ", test_pan_array.shape)

             print("Saving the file...")
             pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
             print("Closing the file...")
             f.close()

             print("")
             print("The dataset has been saved and it is ready for the training! \n")
             print("")

        except Exception as e:
             print('Unable to save data to', pickle_file, ':', e)
             raise

##
# Given a pickle file name and an element number it show the element
# and the associated pan-tilt labels.
# @param pickle_file path to the pickle file
# @param element an integer that specifies which element to return
# @param element_type the dataset to acces (training or test)
# @param img_size the size of the image (default 64x64 pixels)
def show_pickle_element(pickle_file, element, element_type="training", img_size=64):

    #Check if the file exists
    if os.path.isfile(pickle_file) == False:
        print("The pickle file do not exist: " + pickle_file)
        raise ValueError('Error: the pickle file do not exist.')

    #Open the specified dataset and return the element
    if(element_type == "training"):
        with open(pickle_file, 'rb') as f:
            handle = pickle.load(f)
            training_dataset = handle['training_dataset']
            training_tilt_label = handle['training_tilt_label']
            training_pan_label = handle['training_pan_label']
            del handle  # hint to help gc free up memory
            print("Selected element: " + str(element))
            print("Tilt: " + str(training_tilt_label[element]))
            print("Pan: " + str(training_pan_label[element]))
            print("")
            img = training_dataset[element]
            img = np.reshape(img, (img_size,img_size,3))
            cv2.imwrite( "./image.jpg", img );
            #cv2.imshow('image',img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

    elif(element_type == "test"):
            handle = pickle.load(f)
            test_dataset = handle['test_dataset']
            test_tilt_label = handle['test_tilt_label']
            test_pan_label = handle['test_pan_label']
            del handle  # hint to help gc free up memory
            print("Selected element: " + str(element))
            print("Tilt: " + str(test_tilt_label[element]))
            print("Pan: " + str(test_pan_label[element]))
            print("")
            img = test_dataset[element]
            img = np.reshape(img, (img_size,img_size,3))
            cv2.imwrite( "./image.jpg", img );
            #cv2.imshow('image',img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

    else:
        raise ValueError('Error: element_type must be training or test.')


##
# To use the function reported below you have to and add the right folder paths.
#
#
def main():

    #1- It creates the CSV file and cropped/resized faces
    # First of all you have to specify where the uncompressed folder with the dataset is located
    # Specify an output folder and the image size (be careful to choose this size, it must be less
    # than the dimension of the original faces). You can choose if save the image in grayscale or colours.

    #create_csv(input_path="./", output_path="./output", img_size=64, colour=True, normalisation=False)


    #2- It creates 15 pickle files containing numpy arrays with images and labels.
    # You have to specify the CSV file path created in step 1.

    #create_loo_pickle(csv_path="./prima_label.csv", output_path="./output", shuffle=False)


    #3- You can check that everything is fine using this function.
    # In this example it takes a random element and save it in the current folder.
    # It prints the Pan and Tilt labels of the element.
    #element = np.random.randint(2600)

    #show_pickle_element(pickle_file="./output/prima_p1_out.pickle", element=element, element_type="training", img_size=64)


if __name__ == "__main__":
    main()
