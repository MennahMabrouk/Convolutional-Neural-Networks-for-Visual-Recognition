'''
This code imports the necessary libraries, including TensorFlow, NumPy, and Keras, 
and then defines some variables that are used throughout the code.

It then loads a dataset of images, splits it into training and validation sets, 
and applies a scaling function to the images in both sets. The scaling function divides 
each pixel value in the images by 255 to normalize the pixel values to the range [0,1].

Next, the code defines two functions for processing the input and target images. 
The process_input function converts the input image to the YUV color space, splits the image 
into its Y (luminance) channel and its U and V (chrominance) channels, and then resizes 
the Y channel to the specified input size. The process_target function simply converts the target image to 
the YUV color space and returns the Y channel.

The training and validation sets are then mapped to these two functions 
using a lambda function, which allows them to be applied to each image in the dataset. 
The datasets are also pre-fetched for better performance during training.

Finally, the code defines a function get_model that creates a convolutional neural network model 
for image super-resolution. The model takes an input image and applies a series of convolutional layers to it, followed 
by a final depth-to-space layer that upscales the image by the specified factor. 
The model architecture is based on the "ESPCN" model for super-resolution.
'''
import tensorflow as tf             # Importing TensorFlow library
import os                            # Importing Operating System module
import numpy as np                   # Importing NumPy library
from tensorflow import keras        # Importing Keras from TensorFlow
from tensorflow.keras import layers # Importing Layers module from Keras
from tensorflow.keras.preprocessing.image import load_img, array_to_img, img_to_array   # Importing image pre-processing modules from Keras
from tensorflow.keras.preprocessing import image_dataset_from_directory   # Importing image dataset creation module from Keras
from IPython.display import display  # Importing display module from IPython

# Download dataset from url and extract data directory path
dataset_url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
data_dir = keras.utils.get_file(origin=dataset_url, fname="BSR", untar=True)
root_dir = os.path.join(data_dir, "BSDS500/data")

# Set variables for image cropping, upscaling, and batch size
crop_size = 300         # Setting the crop size of images
upscale_factor = 3      # Setting the scale factor for image upscaling
input_size = crop_size // upscale_factor  # Calculating input size of images after downscaling
batch_size = 8          # Setting the batch size for training dataset

'''
This code creates training and validation datasets using the image_dataset_from_directory function and performs pre-processing on them
'''

# Load training dataset from directory
train_ds = image_dataset_from_directory(
    root_dir,
    batch_size=batch_size,
    image_size=(crop_size, crop_size),
    validation_split=0.2,
    subset="training",
    seed=1337,
    label_mode=None,
)

# Load validation dataset from directory
valid_ds = image_dataset_from_directory(
    root_dir,
    batch_size=batch_size,
    image_size=(crop_size, crop_size),
    validation_split=0.2,
    subset="validation",
    seed=1337,
    label_mode=None,
)

# Function for scaling input image pixels to between 0 and 1
def scaling(input_image):
    input_image = input_image / 255.0  # Scaling the input image to a range of [0,1]
    return input_image

train_ds = train_ds.map(scaling)  # Applying scaling to training dataset
valid_ds = valid_ds.map(scaling)  # Applying scaling to validation dataset

dataset = os.path.join(root_dir, "images")
test_path = os.path.join(dataset, "test")

test_img_paths = sorted(
    [
        os.path.join(test_path, fname)
        for fname in os.listdir(test_path)
        if fname.endswith(".jpg")
    ]
)

'''
This code defines functions for processing the input and target images, maps those functions to the training and validation datasets
'''

def process_input(input, input_size, upscale_factor):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return tf.image.resize(y, [input_size, input_size], method="area")

def process_target(input):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return y

train_ds = train_ds.map(
    lambda x: (process_input(x, input_size, upscale_factor), process_target(x))
)
train_ds = train_ds.prefetch(buffer_size=32)

valid_ds = valid_ds.map(
    lambda x: (process_input(x, input_size, upscale_factor), process_target(x))
)
valid_ds = valid_ds.prefetch(buffer_size=32)

def get_model(upscale_factor=3, channels=1):
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(64, 5, **conv_args)(inputs)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(32, 3, **conv_args)(x)
    x = layers.Conv2D(channels * (upscale_factor**2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    return keras.Model(inputs, outputs)
