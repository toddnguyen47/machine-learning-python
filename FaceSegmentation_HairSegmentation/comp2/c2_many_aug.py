from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Lambda, Input  # importing the layers that will be used for our model
from keras.layers.merge import concatenate  # concatenate 2 layers together
from keras.models import Model  # to build a sequential model
import numpy as np  # numpy is life
import tensorflow as tf  # to use the mean_iou metrics
from keras import backend as K  # to use the mean_iou metrics
from keras import callbacks  # to use ModelCheckpoint and LearningRateScheduler
from keras.optimizers import Adam  # optimizier that will be used
from skimage import transform  # to resize images
from imageio import imread, imsave  # to read in images and to save some test images
import os, time, random  # utility libraries that will be used
from scipy.ndimage.interpolation import zoom  # zoom function for image manipulation

img_size = 256  # image size that will be used. need to be a power of 2
orig_size = 250  # targeted mask size
img_channel = 3  # number of channels to be used. we will use colored train images
num_batchsize = 32  # batch size per epoch
num_epochs = 25  # total number of epochs
num_image_aug_options = 6  # maximum number of image augmentations (one method per image)
num_image_aug2_options = 3  # maximum number of image augmentations two methods per image
script_name = os.path.basename(__file__).split('.')[0]  # get the current script name without the extension

base_path = "/home/tnguy47/CS599-02_MachineLearning/Assignment3/data/"  # base path
train_path = base_path + "training_images/training_images"  # path of training images
face_path_target = base_path + "training_masks_face/training_masks_face"  # path of masks of face

train_path_valid = base_path + "validation_images/validation_images"  # path of validation iamages
face_path_valid = base_path + "validation_masks_face/validation_masks_face"  # path of validation masks of face


# Custom Model Checkpoint to only keep the best 4 weights
class CustomModelCheckpoint(callbacks.ModelCheckpoint):
    # Overwriting the on_epoch_end function
    def on_epoch_end(self, epoch, logs=None):
        cur_epoch = epoch + 1 # get current epoch
        # if current epoch is greater than 3, save the weights
        if (cur_epoch >= 3):
            super(CustomModelCheckpoint, self).on_epoch_end(epoch, logs)  # call the parent ModelCheckpoint's on_epoch_end function
            # Keep only the best 4 weights
            list1 = []  # list to store all weights
            for file1 in os.listdir():  # list all file in current directory
                if os.path.isfile(file1) and "weight" in file1:  # if it is a file and its name contains "weight"
                    list1.append(file1)  # append that file

            list1 = sorted(list1)  # sort the list from lowest loss weights to highest loss weights
            for i in list1[4:]:  # keep the best 4 weights, which will have the lowest val_loss
                print("Removing {}".format(i))  # printing the removal of a weight
                os.remove(i)  # remove weights with more valid_loss


# Detect black only or white only images
def detect_black_white_only(img):
    mean = img.mean()  # get the image's mean
    std = img.std()  # get the image's standard deviation
    range1 = 70  # range starting from the middle pixel value
    black_threshold = 127 - range1  # upper limit for black_only mean
    white_threshold = 127 + range1  # lower limit for white_only mean
    std_threshold = 8  # upper limit for standard deviation
    black_only = False  # initialize boolean value as false`
    white_only = False  # initialize boolean value as false`
    if (mean < black_threshold) and (std <= std_threshold):  # check the image's mean against thresholds
        black_only = True  # if the image's mean and std is below the threshold, it is a black only image
    if (mean > white_threshold) and (std <= std_threshold):  # check the image's mean against thresholds
        white_only = True  # if the image's mean is greater than the white_threshold and the standard deviation is greater than std threshold, it is a white only image

    return black_only, white_only  # return the 2 boolean values


# Get the output name for the output folder containing the image masks
def get_output_name_no_ext():
    file_count = 0  # initial number
    outfile = script_name + str(file_count)  # initial folder name
    while (os.path.isdir(outfile)):  # check if the folder already exists
        file_count += 1  # if folder exists, increment the counter by 1
        outfile = script_name + str(file_count)  # check again if folder exists

    return outfile  # return the name of the folder


# Same as get_output_name_no_ext, except we are appending "output" at the start
def get_output_dir():
    file_count = 0  # initial number
    outfile = "./output_" + str(file_count)  # initial folder name
    while (os.path.isdir(outfile)):  # check if the folder already exists
        file_count += 1  # if folder exists, increment the counter by 1
        outfile = "./output_" + str(file_count)  # check again if folder exists

    return outfile  # return the name of the folder


# Fancy printing at the end of a batch job
def print_ending(outfile, end_time):
    print("")
    print("//**************************************************")
    print("//   {} output".format(outfile))
    print("//   Time finished: {}".format(end_time))
    print("//**************************************************")
    print("")


# Augmenting the image, one augmentation per image
def img_aug(image, label, option, angle=20, resize_rate=0.9, shear_angle=20, zoom_amt=0.80):
    optionRotateCW = 0  # Rotate Clockwise
    optionShearCW = 1  # Shear clockwise
    optionZoom = 2  # zoom
    optionHFlip = 3  # Horizontal flip
    optionRotateCCW = 4  # rotate counter clockwise
    optionShearCCW = 5  # shear counter clockwise

    # Error checking. If passing in an incorrect option, exit the program
    if (option >= num_image_aug_options):
        print("Incorrect augmentation option")
        exit(-1)

    # Reference: https://www.kaggle.com/shenmbsw/data-augmentation-and-tensorflow-u-net/code
    # Rotate Clockwise
    if (option == optionRotateCW):
        rotate_angle = (random.random() * angle) / 180.0 * np.pi  # from -angle to angle
        # Create Affine transform
        affine_tf = transform.AffineTransform(rotation=rotate_angle)
        # Apply transform to image data
        image = transform.warp(image, inverse_map=affine_tf, mode='reflect')
        label = transform.warp(label, inverse_map=affine_tf, mode='reflect')
    # Shear Clockwise
    elif (option == optionShearCW):
        sh = (random.random() * shear_angle) / 180.0 * np.pi  # from -shear_angle to shear_angle
        # Create Affine transform
        affine_tf = transform.AffineTransform(shear=sh)
        # Apply transform to image data
        image = transform.warp(image, inverse_map=affine_tf, mode='reflect')
        label = transform.warp(label, inverse_map=affine_tf, mode='reflect')
    # zoom in image
    elif (option == optionZoom):
        image = zoom(image, zoom_amt, mode="reflect")  # use scipy.ndimage.interpolation's zoom function
        image = image[:, :, 0]  # since the zoomed image will be black and white, we need to extract the first dimension
        image = np.stack((image, image, image), axis=2)  # then restack the dimensions to make a 3 dimensional image
        image = transform.resize(image, (img_size, img_size, img_channel), mode="reflect")  # Resize the image back to img_size

        label = zoom(label, zoom_amt, mode="reflect")  # do the same transformation to the mask
        label = transform.resize(label, (img_size, img_size, 1), mode="reflect")  # do the same transformation to the mask
    # Horizontal flip
    elif (option == optionHFlip):
        image = image[:, ::-1, :]  # flip the image horizontally using numpy's index slicing
        label = label[:, ::-1, :]  # flip the image horizontally using numpy's index slicing
    # Rotate CCW
    elif (option == optionRotateCCW):
        rotate_angle = -1 * (random.random() * angle) / 180.0 * np.pi  # from -angle to angle
        # Create Affine transform
        affine_tf = transform.AffineTransform(rotation=rotate_angle)
        # Apply transform to image data
        image = transform.warp(image, inverse_map=affine_tf, mode='reflect')
        label = transform.warp(label, inverse_map=affine_tf, mode='reflect')
    # Shear Clockwise
    elif (option == optionShearCCW):
        sh = -1 * (random.random() * shear_angle) / 180.0 * np.pi  # from -shear_angle to shear_angle
        # Create Affine transform
        affine_tf = transform.AffineTransform(shear=sh)
        # Apply transform to image data
        image = transform.warp(image, inverse_map=affine_tf, mode='reflect')
        label = transform.warp(label, inverse_map=affine_tf, mode='reflect')

    return image, label  # return the transformed image and mask


# Apply two types of augmentation simultaneously to an image
def img_aug2(image, label, option, angle=20, resize_rate=0.9, shear_angle=20, zoom_amt=0.80):
    # will only do these 3 combinations
    optionRotateShear = 0
    optionRotateFlip = 1
    optionShearFlip = 2

    # If incorrect option, exit the program
    if (option >= num_image_aug2_options):
        print("Incorrect number of options for img_aug2")
        exit(-1)

    angle_range = angle << 1  # angle * 2; using this for positive and negative angle rotation
    shear_range = shear_angle << 1  # shear_angle * 2; using this for positive and negative shear angle

    # Rotate, then shear
    if (option == optionRotateShear):
        rotate_angle = ((random.random() * angle_range) - angle) / 180.0 * np.pi  # from -(angle) to angle
        sh = ((random.random() * shear_range) - shear_angle) / 180.0 * np.pi  # from -shear_angle to shear_angle
        # Create Affine transform
        affine_tf = transform.AffineTransform(shear=sh, rotation=rotate_angle)
        # Apply transform to image data
        image = transform.warp(image, inverse_map=affine_tf, mode='reflect')
        label = transform.warp(label, inverse_map=affine_tf, mode='reflect')
    # Rotate, then flip horizontally
    elif (option == optionRotateFlip):
        rotate_angle = ((random.random() * angle_range) - angle) / 180.0 * np.pi  # from -(angle) to angle
        # Create Affine transform
        affine_tf = transform.AffineTransform(rotation=rotate_angle)
        # Apply transform to image data
        image = transform.warp(image, inverse_map=affine_tf, mode='reflect')
        label = transform.warp(label, inverse_map=affine_tf, mode='reflect')

        image = image[:, ::-1, :]  # horizontally flip the image
        label = label[:, ::-1, :]  # horizontally flip the mask
    # Shear, then flip
    elif (option == optionShearFlip):
        sh = ((random.random() * shear_range) - shear_angle) / 180.0 * np.pi  # from -shear_angle to shear_angle
        # Create Affine transform
        affine_tf = transform.AffineTransform(shear=sh)
        # Apply transform to image data
        image = transform.warp(image, inverse_map=affine_tf, mode='reflect')
        label = transform.warp(label, inverse_map=affine_tf, mode='reflect')

        image = image[:, ::-1, :]  # horizontally flip the image
        label = label[:, ::-1, :]  # horizontally flip the mask

    return image, label  # return the transformed image and mask


# Load in images while also performing image augmentation
def load_images_image_aug(train_path, train_prefix, mask_path, mask_prefix):
    total_files = 0  # get total number of files
    for file in os.listdir(train_path):  # for every file in train_path
        if ".jpg" in file:  # if there is a ".jpg" in the file name
            total_files += 1  # increment total_files counter

    probability = 0.90  # the probability that a black_only or white_only mask will be included in our dataset
    train_images = []  # list of train images
    mask_images = []  # list of mask images
    skipped_counter = 0  # count number of skipped images. purely for curiosity
    for i in range(1, total_files + 1):  # for loop time!
        mask_name = mask_prefix + "_" + str(i) + ".jpg"  # the name of the mask
        temp_mask = os.path.join(mask_path, mask_name)  # using os.path.join for smart path string joining
        # Load in target image
        img_mask = imread(temp_mask)  # read in the img_mask
        black_only, white_only = detect_black_white_only(img_mask)  # detect if the mask is black_only or white_only
        # Only analyze data if the mask image is not all black only or white only
        # 10% chance that a black_only or white_only mask goes through
        image_skipped = (black_only or white_only) and (random.random() < probability)
        if not image_skipped:  # if the mask was not skipped
            img_mask = transform.resize(img_mask, (img_size, img_size, 1), mode="reflect")  # only transform mask if it is not skipped. Need the mask to have a dimension of 1 for binary cross entropy
            img_name = train_prefix + "_" + str(i) + ".jpg"  # get the train image's name
            temp_train = os.path.join(train_path, img_name)  # smart path joining to get the full path of image
            img_train = imread(temp_train)  # read in the images
            img_train = transform.resize(img_train, (img_size, img_size, img_channel), mode="reflect")  # resize the image
            imsave('test.jpg', (img_train * 255).astype(np.uint8))  # saving the image to make sure it looks okay

            train_images.append(img_train)  # append image to images list
            mask_images.append(img_mask)  # append mask to image mask

            # Apply one type of transformation to images
            for jj in range(num_image_aug_options):  # loop through all available options
                img2, mask2 = img_aug(image=img_train, label=img_mask, option=jj)  # augment the image and the mask
                train_images.append(img2)  # append the newly transformed image to image list
                mask_images.append(mask2)  # append the newly transformed mask to mask list
            # Apply 2 types of transformation to images simultaneously; also, apply ALL transformations once
            for jj in range(num_image_aug2_options):
                img2, mask2 = img_aug(image=img_train, label=img_mask, option=jj)
                train_images.append(img2)  # append the newly transformed image to image list
                mask_images.append(mask2)  # append the newly transformed mask to mask list
        else:
            skipped_counter += 1  # increment skipped counter

        # Printing the current resizing progress
        print('Resizing: {}/{}, {:.2%}'.format(i, total_files, (i) / total_files), end="\r")

    print("")  # print a new line
    print("Did not include {} images".format(skipped_counter))  # see how many masks were skipped
    return train_images, mask_images  # return the list of train images and mask images


# Define IoU metric
# Reference: https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277?scriptVersionId=2164855/code
def mean_iou(y_true, y_pred):
    prec = []  # keep track of all the score
    for t in np.arange(0.5, 1.0, 0.05):  # random range from 0.5 to 1 with increment of 0.05
        y_pred_ = tf.to_int32(y_pred > t)  # if y_pred is greater than t
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)  # call tensorflow's mean_iou function
        K.get_session().run(tf.local_variables_initializer())  # run tf's lcoal variables initializer
        with tf.control_dependencies([up_opt]):  # open tf's control_dependencies
            score = tf.identity(score)  # get the score
        prec.append(score)  # append the score
    return K.mean(K.stack(prec), axis=0)  # return the mean


# Reference: https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
# Reference: https://github.com/akirasosa/mobile-semantic-segmentation/blob/master/learning_rate.py
def lrn_rate_decay(epoch):
    # Last 15%
    if epoch > num_epochs * 0.85:
        learn_rate = 0.00001
    # Next 55%
    elif epoch > num_epochs * 0.30:
        learn_rate = 0.0001
    # First 30%
    else:
        learn_rate = 0.001

    print("learning rate: {}".format(learn_rate))  # print the current learning rate
    return learn_rate  # return new learning rate


# Reference: https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277?scriptVersionId=2164855/code
def custom_model():
    inputs = Input((img_size, img_size, img_channel))  # define the input layer
    s = Lambda(lambda x: x)(inputs)  # lambda x : x means do not do any adjustments to x (since transform() already normalizes images from 0 to 1)

    # Build U-Net model
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(s)  # Conv2D layer with 32 filters and same padding. "same" results in padding the input such that the output has the same length as the original input
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)  # Conv2D layer with 32 filters
    p1 = MaxPooling2D((2, 2))(c1)  # Max pooling layer

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)  # Conv2D layer with 64 filters
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)  # Conv2D layer with 64 filters
    p2 = MaxPooling2D((2, 2))(c2)  # Max pooling layer

    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)  # Conv2D layer with 128 filters
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)  # Conv2D layer with 128 filters
    p3 = MaxPooling2D((2, 2))(c3)  # Max pooling layer

    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)  # Conv2D layer with 256 filters
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4)  # Conv2D layer with 256 filters
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)  # Max pooling layer

    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(p4)  # bottom of unet. Conv2D layer with 512 filters
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(c5)  # Conv2D layer with 512 filters

    u6 = Conv2DTranspose(img_size, (2, 2), strides=(2, 2), padding='same')(c5)  # time to go back up! Conv2D Transpose layer with 64 output space
    u6 = concatenate([u6, c4])  # concatenate u6 layer with c4 layer
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(u6)  # Conv2D layer with 256 filters
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(c6)  # Conv2D layer with 256 filters

    u7 = Conv2DTranspose(img_size >> 1, (2, 2), strides=(2, 2), padding='same')(c6)  # Conv2DTranspose layer with 32 output space
    u7 = concatenate([u7, c3])  # concatenate layer u7 with c3
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(u7)  # Conv2D layer with 128 filters
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(c7)  # Conv2D layer with 128 filters

    u8 = Conv2DTranspose(img_size >> 2, (2, 2), strides=(2, 2), padding='same')(c7)  # Conv2DTranspose layer with 16 output space
    u8 = concatenate([u8, c2])  # concatenate u8 with c2
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(u8)  # Conv2D layer with 64 filters
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(c8)  # Conv2D layer with 64 filters

    u9 = Conv2DTranspose(img_size >> 3, (2, 2), strides=(2, 2), padding='same')(c8)  # Conv2DTranspose layer with 8 output space
    u9 = concatenate([u9, c1], axis=3)  # concatenate u9 with c2
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(u9)  # Conv2D layer with 32 filters
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(c9)  # Conv2D layer with 32 filters

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)  # action layer, using sigmoid for binary classification

    model = Model(inputs=[inputs], outputs=[outputs])  # make the sequential model
    print(model.summary())  # print out the model's summary

    return model  # return the model

###############################################################################
#  End function declaration
###############################################################################


# Move any leftover saved weights
backup_weights_output = './backup_weights/'  # path of backup folder
if not (os.path.isdir(backup_weights_output)):  # if directory does not exist
    os.mkdir(backup_weights_output)  # create the directory
for file1 in os.listdir():  # for every file in current directory
    if os.path.isfile(file1) and "weight" in file1:  # if the file is a file and its name contains "weight"
        os.rename(file1, os.path.join(backup_weights_output, file1))  # move to the backup weights folder

# Create the model
model = custom_model()  # create the U-net model

train_data, train_target = load_images_image_aug(train_path, "train_img", face_path_target, "train_mask")  # load in images and mask + their augmentations
print("Len of train_data: " + str(len(train_data)))  # printing the total number of images + augmented images

# Load in the validation images and validation masks
train_valid, train_valid_target = load_images_image_aug(train_path_valid, "validation_img",
                                                        face_path_valid, "validation_mask")

# Convert to numpy arrays
print("Converting train_data to numpy array...")
train_data = np.array(train_data)  # Convert to numpy arrays
print("Converting train_target to numpy array...")
train_target = np.array(train_target)  # Convert to numpy arrays
print("Converting train_valid to numpy array...")
train_valid = np.array(train_valid)  # Convert to numpy arrays
print("Converting train_valid_target to numpy array...")
train_valid_target = np.array(train_valid_target)  # Convert to numpy arrays

# Standardizing data
# Reference: https://github.com/akirasosa/mobile-semantic-segmentation/blob/master/data.py
mean = train_data.mean()  # get the mean of all train data images
std = train_data.std()  # get the standard deviation of all mask images
train_data_mean = np.array([[[mean, mean, mean]]])  # convert the mean to a 3D numpy array. Need to be 3D for subtraction
train_data_std = np.array([[[std, std, std]]])  # convert the std deviation to a 3D numpy array
train_data = (train_data - train_data_mean) / (train_data_std + 1e-7)  # for all images, subtract the mean and divide by the standard deviation. Add 1e-7 to prevent dividing by zero

# same as above, but now we are standardizing the validation data
mean = train_valid.mean()  # get the mean
std = train_valid.std()  # get standard deviation
train_data_mean = np.array([[[mean, mean, mean]]])  # convert to 3D numpy array
train_data_std = np.array([[[std, std, std]]])  # convert to 3D numpy array
train_valid = (train_valid - train_data_mean) / (train_data_std + 1e-7)  # for all images, subtract the mean and divide by the standard deviation. Add 1e-7 to prevent dividing by zero

# If train mask is 2D only, expand it to a third dimension
# This is needed so that it can be a proper target mask
if (len(train_target.shape) < 4):
    train_target = np.expand_dims(train_target, axis=3)
if (len(train_valid_target.shape) < 4):
    train_valid_target = np.expand_dims(train_valid_target, axis=3)

# Printing out progress
print('Data: {}, Target: {}'.format(train_data.shape, train_target.shape))
print('Data type: {}, {}'.format(train_data.dtype, train_target.dtype))

# Compile the model using Adam optimizer with a lr of 0.001,
# binary_crossentropy as the loss, and the custom mean_iou as the metrics
model.compile(
    optimizer=Adam(lr=0.001),
    loss='binary_crossentropy',
    metrics=[mean_iou]
)

model_check_filepath = "weights-{val_loss:.4f}-{epoch:02d}.hdf5"  # get the weights name. need the loss to be first for sorting
# Get a custom mdoel checkpoint so that it will only save the best 4 weights
# save_best_only need to be false so we can save all weights. We will filter these weights using our custom function
model_check = CustomModelCheckpoint(model_check_filepath, save_best_only=False, save_weights_only=True, verbose=0)
# Get the callbacks for a custom learn_rate_decay
learn_rate_decay = callbacks.LearningRateScheduler(lrn_rate_decay)

# Fit the model with model_check and learn_rate_decay as callbacks
return_model = model.fit(
    train_data, train_target,
    batch_size=num_batchsize, epochs=num_epochs,
    verbose=1, validation_data=(train_valid, train_valid_target),
    callbacks=[model_check, learn_rate_decay]
)

# Printing out the end time
end_time = time.ctime()
print_ending(get_output_name_no_ext(), end_time)
