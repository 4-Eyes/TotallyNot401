'''
Brent Martin 22/4/2016
Example code for running a convolutional network on the cacophony images.
Adapted from convolutional.py from the deep learning network:

http://deeplearning.net/tutorial/lenet.html

See convolutional_mlp.py for the implementation of a 
convolutional + max pooling layer using theano.

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
'''

from __future__ import print_function

import getpass
import os
import sys
import timeit
import recording

import numpy

import theano
import theano.tensor as T

from mlp import HiddenLayer
from logistic_sgd import LogisticRegression
from convolutional_mlp import LeNetConvPoolLayer

from data import load_data
from data import get_metadata

# for plotting images of the filters
from PIL import Image
import pylab

im_width = None
im_height = None


def new_size(dimensions, filter_size):
    return (dimensions[0] - filter_size + 1) // 2, (dimensions[1] - filter_size + 1) // 2


def evaluate_lenet5(learning_rate, n_epochs, nkerns, batch_size, use_foreground):
    """ Demonstrates lenet on a small sample of the cacophony dataset
	using a network consisting of: 
	- two (convolutional + max pool) layers
	- one fully connected hidden layer
	- logistic regression to determine the final class from the hidden layer outputs

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: list of ints
    :param nkerns: number of kernels in each layer
	
	Adapted from convolutional_mlp::evaluate_lenet5
    """
    global im_width, im_height

    filter_size = 5  # number of pixels across for the convolutional filter

    rng = numpy.random.RandomState(23455)  # Use this one for the same result each time
    # rng = numpy.random.RandomState()

    datasets = load_data(0.3, use_foreground)
    im_width, im_height = get_metadata()[1]

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.vector('y', "int32" if getpass.getuser() == "Matthew" else "int64")  # the labels are presented as 1D vector of
    # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 48 * 64)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (48, 64) is the size of cacophony small images. (height, width)
    layer0_input = x.reshape((batch_size, 1, im_height, im_width))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (48-5+1 , 64-5+1) = (44, 60)
    # maxpooling reduces this further to (44/2, 60/2) = (22, 30)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 22, 30)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, im_height, im_width),
        filter_shape=(nkerns[0], 1, filter_size, filter_size),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (22-5+1, 30-5+1) = (18, 26)
    # maxpooling reduces this further to (18/2, 26/2) = (9, 13)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 9, 13)
    layer1_dim = new_size((im_height, im_width), filter_size)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], layer1_dim[0], layer1_dim[1]),
        # previous layer generated 22*30 sized "images"
        filter_shape=(nkerns[1], nkerns[0], filter_size, filter_size),
        poolsize=(2, 2)
    )
    layer2_dim = new_size(layer1_dim, filter_size)
    layer2 = LeNetConvPoolLayer(
        rng,
	input=layer1.output,
	image_shape=(batch_size, nkerns[1], layer2_dim[0], layer2_dim[1]),
	filter_shape=(nkerns[2], nkerns[1], filter_size, filter_size),
	poolsize=(2,2)
    )
    layerA_dim = new_size(layer2_dim, filter_size)
    layerA = LeNetConvPoolLayer(
	rng,
	input=layer2.output,
	image_shape=(batch_size, nkerns[2], layerA_dim[0], layerA_dim[1]),
	filter_shape=(nkerns[3], nkerns[2], filter_size, filter_size),
	poolsize=(2,2)
    )
    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 9 * 13),
    # or (1, 50 * 9 * 13) with the default values.
    layer3_input = layerA.output.flatten(2)
    layer3_dim = new_size(layerA_dim, filter_size)
    # construct a fully-connected sigmoidal layer
    layer3 = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=nkerns[2] * layer3_dim[0] * layer3_dim[1],
        # 9*13 is the number of pixels in the "image" from the previous layer
        n_out=batch_size,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(input=layer3.output, n_in=batch_size,
                                n_out=get_metadata()[0])  # n_out is the number of classes

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params + layerA.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
        ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
    # go through this many
    # minibatches before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    test_score = 1
    save_file_name = str(n_epochs) + str(learning_rate) + str(batch_size) + str(nkerns) + str(use_foreground) + ".csv"
    recording.append_to_file(save_file_name, "epoch, minibatch, validation error, test error,")
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.0))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                        ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
                    recording.append_to_file(save_file_name, "{0},{1}/{2},{3},{4},".format(epoch, minibatch_index+1, n_train_batches, this_validation_loss * 100.0, test_score * 100.0))
                else:
                    recording.append_to_file(save_file_name, "{0},{1}/{2},{3},{4},".format(epoch, minibatch_index+1, n_train_batches, this_validation_loss * 100.0, 0))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    recording.append_to_file(save_file_name, "end time = {0}".format(end_time))
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    datasets = None
    # display_output(test_set_x, batch_size, layer0, nkerns[0])

    # display the final filters for the convolutional layers
    # display_conv_filters("Layer 0", layer0)
    # display_conv_filters("Layer 1", layer1)


def display_output(images, batch_size, layer, num_feature_maps):
    '''
	Visualises the convolution of the last image to be processed.
	NOTE: SAMPLE CODE ONLY - ONLY USED FOR LAYER0
	'''
    global im_height, im_width
    # Create a theano function that computes the layer0 output for a single batch
    # This declares to theano what the input source and output expression are
    f = theano.function([layer.input], layer.output)

    # recast the inputs from (batch_size, num_pixels) to a 4D tensor of size (batch_size, 1, height, width)
    # as expected by the convolutional layer (the 1 is the "depth" of the input layer)
    img = images.eval()[0: batch_size].reshape(batch_size, 1, im_height, im_width)
    filtered_img = f(img)
    filtered_img = numpy.add(filtered_img, -1. * filtered_img.min())  # Avoid negatives by ensuring the min value is 0

    pylab.gray();

    # Plot the original image
    pylab.subplot(1, 4, 1);
    pylab.axis('off');
    pylab.imshow(img[0, 0, :, :])
    pylab.title("Original image")

    # Plot each feature map
    for map_num in range(num_feature_maps):
        pylab.subplot(1, num_feature_maps + 1, map_num + 2);
        pylab.axis('off');
        pylab.imshow(filtered_img[0, map_num, :, :])
        pylab.title("Feature map " + str(map_num))
    pylab.show()


def display_conv_filters(title, layer):
    '''
    displays the filters as "images"
    - one row per feature map in this layer
    - one column per input into this layer (one for the first layer, 
      one per previous layer's feature maps for the next layer)
    '''
    filters = layer.W  # 4D Tensor of dimensions <number of feature maps, number of inputs, height, width>
    bias = layer.b  # vector of biases, one per feature map

    pylab.gray()  # make plots grayscale

    i = 0
    num_feature_maps = len(filters.eval())
    for map_num in range(num_feature_maps):  # iterate through the feature maps
        num_inputs = len(filters.eval()[map_num])
        for input_num in range(num_inputs):  # iterate through the inputs to this feature map
            i += 1
            img_data = filters.eval()[map_num][input_num]  # extract the (array of) filter values from the tensor slice
            pylab.subplot(num_feature_maps, num_inputs, i);
            pylab.axis('off');
            pylab.imshow(img_data)  # Plot it
            if (i == 1): pylab.title(title)
    pylab.show()


if __name__ == '__main__':
    evaluate_lenet5(learning_rate=0.01, n_epochs=200, nkerns=[8,4,2,2],batch_size=100,use_foreground=False)
    
#    for i in range(3, 8):
#        evaluate_lenet5(learning_rate=0.01,
#                        n_epochs=100,
#                        nkerns=[i, i],  # number of units in each convolutional layer
#                        batch_size=455,
#                        use_foreground=False) # number of rows to process at a time (1 = fully stochastic, n_examples = non-stochastic)
#        evaluate_lenet5(learning_rate=0.01,
#                        n_epochs=100,
#                        nkerns=[i, i],  # number of units in each convolutional layer
#                        batch_size=455,
#                        use_foreground=True) # number of rows to process at a time (1 = fully stochastic, n_examples = non-stochastic)
