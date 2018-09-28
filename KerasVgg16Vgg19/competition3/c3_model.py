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

image_size = 48 # image size to resize to
num_epochs = 20	# number of epochs to run
custom_batch_size = 64 # number of batch size per epoch

base_path = '/home/tnguy47/CS599-02_MachineLearning/Assignment2/data/comp3/' # base path of competition 3's data
npy_train_arr = base_path + 'train_data.csv' # train_data's csv path
train_target = base_path + 'train_target.csv' # train_target's csv path
test_train_arr = base_path + 'test_data.csv' # test_data's csv path

def load_img_from_csv(csv_file, header_i=None, index_col_i=None): # load in images from the csv file
	# read in the csv file while being mindful that there is no header and no index column
	data_input = pd.read_csv(csv_file, header=header_i, index_col=index_col_i)
	np_arr = data_input.as_matrix() # convert the csv input into a numpy array
	np_arr = np_arr.reshape(np_arr.shape[0], image_size, image_size, 1) # reshape the numpy array to reflect the image size
	return np_arr # return the reshaped numpy array

def load_target(path):
	csv_file = pd.read_csv(train_target, header=None, index_col=None) # read in the csv file
	targets = csv_file.as_matrix() # convert the target into a numpy matrix
	return targets # return the converted numpy array

# Detect all black or all white images
def detect_black_white_only(img):
	mean = img.mean() # get the mean of the image
	std = img.std() # get the standard deviation of the image
	black_threshold = 40 # threshold for all black images - used for mean
	white_threshold = 215 # threshold for all white images - used for mean
	std_threshold = 8 # the standard deviation threshold
	black_only = False # set black_only to False initially
	white_only = False # set white only to False initially
	# if the mean is less than the black_threshold and the std dev is less than std dev threshold
	if (mean < black_threshold) and (std <= std_threshold):
		black_only = True # then we set black_only to true

	# if the mean is greater than the white_threshold and the std dev is less than std dev threshold
	if (mean > white_threshold) and (std <= std_threshold):
		white_only = True # then we set white_only to true

	return black_only, white_only # return the determined boolean for black_only and white_only


# Detect black only or white only images
def c3_preprocessing(train_data, train_target):
	index_to_del = [] # the list of indices to delete
	for i in range(train_data.shape[0]): # iterate through all the train_data images
		black_only, white_only = detect_black_white_only(train_data[i]) # use the function detect_black_white_only to detect images
		if black_only or white_only: # if we found that the image is black_only or white_only
			index_to_del.append(i) # add the index to the list

	num_deletes = 0 # track the number of deletes
	for index in over_max_age: # iterate over all indices in over_max_age
		temp_index = index - num_deletes # index will be the index stored in over_max_age minus num_deletes
										 # since the array after deletion will shift its indices by 1
		if (temp_index < 0):
			temp_index = 0 # to make sure we don't run into an array index error

		train_data = np.delete(train_data, [temp_index], axis=0) # delete train_data at index temp_index at axis = 0
		train_target = np.delete(train_target, [temp_index], axis=0) # delete train_target at index temp_index at axis = 0

		num_deletes += 1 # increment num_deletes
		print('\r{}/{}, {:.2%}'.format(num_deletes, len(over_max_age),
		num_deletes/len(over_max_age)), end="") # printing a progress bar

	print("") # printing a newline
	return train_data, train_target # return the trimmed train_data and train_target

def custom_model():
	model = Sequential() # create a Sequential model
	model.add(Conv2D(32,
		kernel_size=(3,3),
		strides=(1,1),
		input_shape=(image_size,image_size,1),
		activation='relu'
	)) # added a Conv2D input layer with input_shape of (48,48,1) and 32 layers
	model.add(MaxPooling2D(pool_size=(2,2))) # created a MaxPooling2D layer with a pool size of (2,2)

	model.add(Conv2D(32, kernel_size=(3,3), activation='relu')) # created a Conv2D layer with 32 filters
	model.add(ZeroPadding2D(padding=(1,1))) # created a ZeroPadding2D layer
	model.add(Conv2D(32, kernel_size=(3,3), activation='relu')) # created a Conv2D layer with 32 filters
	model.add(MaxPooling2D(pool_size=(2,2))) # created a MaxPooling2D layer with a pool size of (2,2)

	model.add(Dropout(0.25)) # added a Dropout layer with a probability of 0.25

	model.add(Conv2D(64, kernel_size=(3,3), activation='relu')) # created a Conv2D layer with 64 filters
	model.add(ZeroPadding2D(padding=(1,1))) # created a ZeroPadding2D layer
	model.add(Conv2D(64, kernel_size=(3,3), activation='relu')) # created a Conv2D layer with 64 filters
	model.add(MaxPooling2D(pool_size=(2,2))) # created a MaxPooling2D layer with a pool size of (2,2)

	model.add(Dropout(0.25)) # added a Dropout layer with a probability of 0.25

	model.add(Conv2D(64, kernel_size=(3,3), activation='relu')) # created a Conv2D layer with 64 filters
	model.add(ZeroPadding2D(padding=(1,1))) # created a ZeroPadding2D layer
	model.add(Conv2D(64, kernel_size=(3,3), activation='relu')) # created a Conv2D layer with 64 filters
	model.add(MaxPooling2D(pool_size=(2,2))) # created a MaxPooling2D layer with a pool size of (2,2)

	model.add(Flatten()) # flatten the above layers
	model.add(Dropout(0.50)) # added a Dropout layer with a probability of 0.50
	model.add(Dense(64, activation='relu')) # added a final Dense hidden layer with 64 filters
	model.add(Dense(3, activation='softmax')) # an output layer that uses softmax for multi-class logistic regression

	return model # return the created model

###############################################################################
##  End function declaration
###############################################################################

model = custom_model() # obtain the created model

train_data = load_img_from_csv(npy_train_arr) # load train data from the csv train file
train_target = load_target(train_target) # load the targets from the csv target file
test_data = load_img_from_csv(test_train_arr) # load the test data from the csv test file

train_data, train_target = c3_preprocessing(train_data, train_target) # detect and delete all black or all white images

train_target = to_categorical(train_target, num_classes=3) # as input to categorical_crossentropy
train_data = train_data / 255 # to normalize input data from 0 to 1

model.compile(
	optimizer='adam',
	loss='categorical_crossentropy',
	metrics=['accuracy']
) # compile the model with adam optimizer and categorical_crossentropy as loss for multi-class classification

return_model = model.fit(train_data, train_target,
	batch_size=custom_batch_size,
	epochs=num_epochs, verbose=1,validation_split=0.2) # train the data with 20% of the train data and train target being used as validation data

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
