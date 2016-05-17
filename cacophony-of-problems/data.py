'''
Brent Martin 22/4/2016
Routines for loading images files into the appropriate data structures 
for processing by the colvolutional network code.
'''
import getpass
import os
import numpy
import theano
import background
import inspect
from theano import tensor as T
from theano.tensor.nnet import conv2d
from PIL import Image
import numpy as np
import pylab

classes = {}
class_iterator = 0
image_size = None

def load_images(train_folder, valid_folder, test_folder, scale, use_foreground):
    ''' 
    Loads folder of imagess for use by convolution_mlp.
    train_set, valid_set, test_set format: tuple(input, target)
    input is a numpy.ndarray of 2 dimensions (a matrix)
    where each row corresponds to an example. target is a
    numpy.ndarray of 1 dimension (vector) that has the same length as
    the number of rows in the input. It should give the target
    to the example with the same index in the input.
    '''
    return load_folder(train_folder, scale, use_foreground), load_folder(valid_folder, scale, use_foreground), load_folder(test_folder, scale, use_foreground),

    
def load_folder(folder, scale, use_foreground):
    '''
    Loads a single folder of images
    - load each image into a vector of greyscale values, making a matrix of <image , pixels>
    - wrap the matrix in a theano shared tensor variable
    - wrap the target vector in a tensor shared variable
    - return as a tuple (images, targets)
    '''
    global class_iterator, classes, image_size
    print("*** Loading folder " + folder + " ***")
    images = []
    clazzes = []
    working_dir = os.path.join(os.getcwd(),folder)
    for dir in [os.path.join(working_dir, d) for d in os.listdir(working_dir) if os.path.isdir(os.path.join(working_dir, d)) and not "disabled" in d]:
        with background.Movement(dir, displayOpenCVImage=False) as m:
            (frames, input_clazzes), image_size, clazzes_unique = m.getMovementImages(scale, use_foreground)

        images.extend(frames)

        for clazz in clazzes_unique:
            if not clazz in classes:
                classes[clazz] = class_iterator
                class_iterator += 1

        for clazz in input_clazzes:
            clazzes.append(classes[clazz])
                
    imagesTensor = theano.shared(numpy.asarray(images))
    targetsTensor = theano.shared(numpy.asarray(clazzes))
    return (imagesTensor, targetsTensor)

def load_data(scale=1.0, use_foreground = False):
    # Change these to point to your training, validation and test set image directories
    if getpass.getuser() == "Matthew":
        TRAIN_DIR = './train'
        VALID_DIR = './valid'
        TEST_DIR = './test'
    else:
        TRAIN_DIR = "/media/james/9a6d3124-40f4-4227-9ef6-5cecdc794447/Reference Images/train"
        VALID_DIR = "/media/james/9a6d3124-40f4-4227-9ef6-5cecdc794447/Reference Images/valid"
        TEST_DIR = "/media/james/9a6d3124-40f4-4227-9ef6-5cecdc794447/Reference Images/test"

    return load_images(TRAIN_DIR, VALID_DIR, TEST_DIR, scale, use_foreground)

def get_metadata():
    global image_size
    return len(classes),image_size