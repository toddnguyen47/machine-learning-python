from keras.models import load_model, Sequential # importing Sequential to create custom models, load_model to load saved models
from keras.layers import Dense, MaxPooling2D, Activation, Conv2D, Flatten, Dropout, ZeroPadding2D # import the required layers
from keras.applications.vgg16 import VGG16, preprocess_input # import VGG16's model to summarize the weights
from keras.preprocessing import image # to use to load in images
from keras import losses, optimizers # import keras' losses and optimizers
import numpy as np # to manipulate numpy arrays
import pandas as pd # to load in csv files
import os # to list directories
import math # to ceil/floor calculations
import time # to get the current time

image_size = 128 # image size to resize to
num_epochs = 15 # number of epochs to run

def load_target(path): # load the target csv file
	csv_file = pd.read_csv(path) # read in the csv file
	target = csv_file['Gender'].tolist() # extract the column 'Gender' and make it into a list
	target = np.array(target) # convert the list to an np array
	return target # return the np array

def custom_model():
	model = Sequential() # create a Sequential layer

	model.add(Conv2D(32,
		kernel_size=(3,3),
		strides=(1,1),
		input_shape=(image_size,image_size,1),
		activation='relu'
	)) # create an Conv2D layer with 32 filters and input_shape of 128,128,1
	model.add(MaxPooling2D(pool_size=(2,2))) # created a MaxPooling2D layer with pool_size of (2,2)

	model.add(Conv2D(32, kernel_size=(3,3), activation='relu')) # created a Conv2D layer with 32 filters
	model.add(ZeroPadding2D(padding=(1, 1))) # Zero pads the image
	model.add(Conv2D(32, kernel_size=(3,3), activation='relu')) # created a Conv2D layer with 32 filters
	model.add(MaxPooling2D(pool_size=(2,2))) # created a MaxPooling2D layer with pool_size of (2,2)

	model.add(Conv2D(64, kernel_size=(3,3), activation='relu')) # created a Conv2D layer with 64 filters
	model.add(MaxPooling2D(pool_size=(2,2))) # created a MaxPooling2D layer with pool_size of (2,2)

	model.add(Conv2D(64, kernel_size=(3,3), activation='relu')) # created a Conv2D layer with 64 filters
	model.add(MaxPooling2D(pool_size=(2,2))) # created a MaxPooling2D layer with pool_size of (2,2)

	model.add(Flatten()) # flatten the previous layers
	model.add(Dropout(0.50)) # added a dropout layer with a probability of 0.50
	model.add(Dense(64, activation='relu')) # created a final Dense hidden layer
	model.add(Dense(1, activation='sigmoid')) # created an output layer with sigmoid activation for two class classification (0 or 1)
	return model

###############################################################################
##  End function declaration
###############################################################################

model = custom_model() # get the model returned from custom_model
print(model.summary()) # print the model's layers

base_path = '/home/tnguy47/CS599-02_MachineLearning/Assignment2/data/comp1and2/' # path of file storage
npy_train_arr = base_path + 'train_io_gray.npy' # path of saved images in a numpy array
test_train_arr = base_path + 'test_io_gray.npy' # path of saved images in a numpy array
train_target = "/home/tnguy47/CS599-02_MachineLearning/Assignment2/competition1/c1_train_target.csv" # path of target csv

train_data = np.load(npy_train_arr) # load in the train images
train_target = load_target(train_target) # load in the train targets
test_data = np.load(test_train_arr) # load in the test images

model.compile(
	optimizer='adam',
	loss='binary_crossentropy',
	metrics=['accuracy']
) # compile the model with adam optimizer, binary crossentropy loss for two-class classification, and accuracy as metrics

model.fit(train_data, train_target, batch_size=100,
epochs=num_epochs, verbose=1,validation_split=0.1) # train the data with batch size of 100, and 10% of train_data and train_target used as validation data

model_name = str(os.path.basename(__file__)).split('.')[0] + ".h5" # get the model's name as this script's name + .h5
model.save_weights(model_name) # save the model's weights

# Predict
print("Predicting...") # Let the user know we are predicting
predictions = model.predict(test_data, verbose=0) # predict test_data outputs

cur_script_name = os.path.basename(__file__) # get this script's filename
cur_script_name = cur_script_name.split('.')[0] + "_output_" # take out the .py and add in _output_
file_count = 0 # to make sure file names do not overlap
outfile = cur_script_name + str(file_count) + ".txt" # file name is script's file name + file_count
while (os.path.isfile(outfile)): # iterate to make sure we do not overwrite any output files
	file_count += 1 # iterate file_count
	outfile = cur_script_name + str(file_count) + ".txt" # trying a new file name

with open(outfile, "w") as file: # writing output to file
	picture_count = 1 # for naming pictures, i.e. train_1, test_1, etc.
	for i in predictions: # iterate over all predictions
		b = str(i[0]) # the current prediction
		file.write("test_" + str(picture_count) + ".jpg," + b + '\n') # writing out the prediction in a csv format
		picture_count += 1 # iterate picture count

end_time = time.ctime()# get the end time
# The 5 lines below are used to determine which output file matches up with which slurm files when
# the user runs batch jobs on bridges.psc.edu
print("")
print("//**************************************************")
print("//   {} output".format(outfile))
print("//   Time finished: {}".format(end_time))
print("//**************************************************")
print("")
