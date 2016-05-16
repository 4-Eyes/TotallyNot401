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
import numpy as np
import pylab

classes = {}
class_iterator = 0
image_size = None

def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    http://stackoverflow.com/questions/7422204/intensity-normalization-of-image-using-pythonpil-speed-issues
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr

def load_images(train_folder, valid_folder, test_folder, scale):
    ''' 
    Loads folder of imagess for use by convolution_mlp.
    train_set, valid_set, test_set format: tuple(input, target)
    input is a numpy.ndarray of 2 dimensions (a matrix)
    where each row corresponds to an example. target is a
    numpy.ndarray of 1 dimension (vector) that has the same length as
    the number of rows in the input. It should give the target
    to the example with the same index in the input.
    '''
    return load_folder(train_folder, scale), load_folder(valid_folder, scale), load_folder(test_folder, scale),

    
def load_folder(folder, scale):
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
        with background.Movement(dir) as m:
            (frames, input_clazzes), image_size, clazzes_unique = m.getMovementImages(scale)

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

    
#def load_image(filename, scale):
#    ''' 
#    Returns the image as a 1d vector of grayscale values
#    '''
#    global image_size

#    with Image.open(filename).convert('L') as img:
#        img.thumbnail((img.size[0] * scale, img.size[1] * scale), Image.ANTIALIAS)
#        if image_size is None:
#            image_size = img.size

#        # extract the image as a vector of grayscale values between 0 and 1
#        raster = (numpy.asarray(img, dtype='float64') / 256.).reshape(image_size[1]*image_size[0])

#        #raster = normalize(np.array(raster))
#    # print raster # DEBUG - shows the vector of greyscale values
#    return raster
    
def load_data(scale=1.0):
    # Change these to point to your training, validation and test set image directories
    TRAIN_DIR = './train'
    VALID_DIR = './valid'
    TEST_DIR = './test'

    return load_images(TRAIN_DIR, VALID_DIR, TEST_DIR, scale)

def get_metadata():
    global image_size
    return len(classes),image_size