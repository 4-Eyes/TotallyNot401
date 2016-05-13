'''
Brent Martin 22/4/2016
Routines for loading images files into the appropriate data structures 
for processing by the colvolutional network code.
'''

import os
import numpy
import theano
import background
import inspect
from theano import tensor as T
from theano.tensor.nnet import conv2d
from PIL import Image

classes = {}
class_iterator = 0
image_size = None

def load_images(train_folder, valid_folder, test_folder):
    ''' 
    Loads folder of imagess for use by convolution_mlp.
    train_set, valid_set, test_set format: tuple(input, target)
    input is a numpy.ndarray of 2 dimensions (a matrix)
    where each row corresponds to an example. target is a
    numpy.ndarray of 1 dimension (vector) that has the same length as
    the number of rows in the input. It should give the target
    to the example with the same index in the input.
    '''
    return load_folder(train_folder), load_folder(valid_folder), load_folder(test_folder),

    
def load_folder(folder):
    '''
    Loads a single folder of images
    - load each image into a vector of greyscale values, making a matrix of <image , pixels>
    - wrap the matrix in a theano shared tensor variable
    - wrap the target vector in a tensor shared variable
    - return as a tuple (images, targets)
    '''
    global class_iterator, classes
    print("*** Loading folder " + folder + " ***")
    images = []
    clazzes = []
    working_dir = os.path.join(os.getcwd(),folder)
    for dir in [os.path.join(working_dir, d) for d in os.listdir(working_dir) if os.path.isdir(os.path.join(working_dir, d))]:
        if "disabled" in dir:
            continue
        m = background.Movement(dir)
        image_paths = m.getMovementImages()

        for clazz, file in image_paths:
            print("...loading " + file)
            images.append(load_image(file))
            if not clazz in classes:
                classes[clazz] = class_iterator
                class_iterator += 1
            clazzes.append(classes[clazz])
    imagesTensor = theano.shared(numpy.asarray(images))
    targetsTensor = theano.shared(numpy.asarray(clazzes))
    return (imagesTensor, targetsTensor)

    
def load_image(filename):
    ''' 
    Returns the image as a 1d vector of grayscale values
    '''
    global image_size

    img = Image.open(filename).convert('L')

    if image_size is None:
        image_size = img.size

    # extract the image as a vector of grayscale values between 0 and 1
    raster = (numpy.asarray(img, dtype='float64') / 256.).reshape(image_size[1]*image_size[0])
    # print raster # DEBUG - shows the vector of greyscale values
    return raster

    
def load_data():
    # Change these to point to your training, validation and test set image directories
    TRAIN_DIR = 'train'
    VALID_DIR = 'valid'
    TEST_DIR = 'test'

    return load_images(TRAIN_DIR, TRAIN_DIR, TRAIN_DIR)

def get_metadata():
    global image_size
    return len(classes),image_size