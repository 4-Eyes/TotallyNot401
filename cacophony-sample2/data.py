'''
Brent Martin 22/4/2016
Routines for loading images files into the appropriate data structures 
for processing by the colvolutional network code.
'''

import os
import numpy
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from PIL import Image


def load_images(train_folder, valid_folder, test_folder, train_targets, valid_targets, test_targets):
	''' 
	Loads folder of imagess for use by convolution_mlp.
	train_set, valid_set, test_set format: tuple(input, target)
	input is a numpy.ndarray of 2 dimensions (a matrix)
	where each row corresponds to an example. target is a
	numpy.ndarray of 1 dimension (vector) that has the same length as
	the number of rows in the input. It should give the target
	to the example with the same index in the input.
	'''
	return load_folder(train_folder, train_targets), load_folder(valid_folder, valid_targets), load_folder(test_folder, test_targets),

	
def load_folder(folder, targets):
	'''
	Loads a single folder of images
	- load each image into a vector of greyscale values, making a matrix of <image , pixels>
	- wrap the matrix in a theano shared tensor variable
	- wrap the target vector in a tensor shared variable
	- return as a tuple (images, targets)
	'''
	print("*** Loading folder " + folder + " ***")
	images = []
	for file in os.listdir(folder):
		print("...loading " + file)
		filename = folder + "/" + file
		images.append(load_image(filename))
	imagesTensor = theano.shared(numpy.asarray(images))
	targetsTensor = theano.shared(numpy.asarray(targets))
	return (imagesTensor, targetsTensor)

	
def load_image(filename):
	''' 
	Returns the image as a 1d vector of grayscale values
	'''
	img = Image.open(filename).convert('L')
	# extract the image as a vector of grayscale values between 0 and 1
	raster = (numpy.asarray(img, dtype='float64') / 256.).reshape(48*64)
	# print raster # DEBUG - shows the vector of greyscale values
	return raster

	
def load_data():
	# Change these to point to your training, validation and test set image directories
	TRAIN_DIR = 'train'
	VALID_DIR = 'valid'
	TEST_DIR = 'test'
	# Sample only - you will need to provide the appropriate target vector here...
	target = [0,0,0,0,1,1,1,1,2,2,2,13]
	return load_images(TRAIN_DIR, VALID_DIR, TEST_DIR, target, target, target)

