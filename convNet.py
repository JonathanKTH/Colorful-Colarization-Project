import numpy as np
import cv2
import tensorflow as tf
import skimage.color as color
import skimage.io as io
from skimage import img_as_ubyte

#NEW
import keras.backend as K

from keras.models import Model, load_model
from keras.callbacks import LambdaCallback
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Conv2DTranspose, Input, BatchNormalization, UpSampling2D
from keras.optimizers import Adam
from keras.engine.topology import Layer
from keras import Sequential

import os, sys

def conv_layer(x, filters, strides=1, idx=1, dilations=1):
    if type(dilations) is int:
        dilations = [dilations]*len(strides)
    elif type(strides) is int:
        strides = [strides]*len(dilations)
    
        
    for i, (stride, dilation) in enumerate(zip(strides, dilations)):
        if type(stride) is int:
            x = Conv2D(filters, 3, strides=stride, padding='same', dilation_rate=dilation,
                       activation='relu', name='conv' + str(idx) + '_' + str(i+1))(x)
        else:
            x = Conv2DTranspose(filters, 3, strides=int(1 / stride), padding='same',
                                activation='relu', name='conv' + str(idx) + '_' + str(i+1))(x)
    return BatchNormalization(name='bn' + str(idx))(x)

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def create_bias(size):
    return tf.Variable(tf.constant(0.1, shape = [size]))

def convolution(inputs, num_channels, filter_size, num_filters):
    weights = create_weights(shape = [filter_size, filter_size, num_channels, num_filters])
    bias = create_bias(num_filters)

  ## convolutional layer
    layer = tf.nn.conv2d(input = inputs, filter = weights, strides= [1, 1, 1, 1], padding = 'SAME') + bias
    layer = tf.nn.tanh(layer)
    return layer


def maxpool(inputs, kernel, stride):
    layer = tf.nn.max_pool(value = inputs, ksize = [1, kernel, kernel, 1], strides = [1, stride, stride, 1], padding = "SAME")
    return layer

def upsampling(inputs):
    layer = tf.image.resize_nearest_neighbor(inputs, (2*inputs.get_shape().as_list()[1], 2*inputs.get_shape().as_list()[2]))
    return layer

if __name__ == "__main__":
    mydir = r'imgs'
    mydirTest = r'imgsTest'
    images = [files for files in os.listdir(mydir)]
    
    imagesTest = [files for files in os.listdir(mydirTest)]
    #print("Hej")
    
    N = len(images)
    data = np.zeros([N, 224, 224, 3]) # N is number of images for training
    for count in range(len(images)):
        img = cv2.resize(io.imread(mydir + '/'+ images[count]), (224, 224))
        data[count,:,:,:] = img
    
    # Test image
    Ntest = len(imagesTest)
    dataTest = np.zeros([Ntest, 224, 224, 3]) # N is number of images for testing
    for count in range(len(imagesTest)):
        img = cv2.resize(io.imread(mydirTest + '/'+ imagesTest[count]), (224, 224))
        dataTest[count,:,:,:] = img
        
    num_train = N
    Xtrain = color.rgb2lab(data[:num_train]*1.0/255)
    xt = Xtrain[:,:,:,0]
    yt = Xtrain[:,:,:,1:]
    yt = yt/128
    xt = xt.reshape(num_train, 224, 224, 1)
    yt = yt.reshape(num_train, 224, 224, 2)
    
    num_test = Ntest
    Xtest = color.rgb2lab(dataTest[:num_test]*1.0/255)
    xtest = Xtest[:,:,:,0]
    xtest = xtest.reshape(num_test, 224, 224, 1)

    session = tf.Session()
    x = tf.placeholder(tf.float32, shape = [None, 224, 224, 1], name = 'x')
    ytrue = tf.placeholder(tf.float32, shape = [None, 224, 224, 2], name = 'ytrue')
    
    
    l_in = Input((224, 224, 1))
    xx = conv_layer(l_in, 64, [1, 2], 1)
    xx = conv_layer(xx, 128, [1, 2], 2)
    xx = conv_layer(xx, 256, [1, 1, 2], 3)
    xx = conv_layer(xx, 512, [1]*3, 4)
    xx = conv_layer(xx, 512, [1]*3, 5, 2)
    xx = conv_layer(xx, 512, [1]*3, 6, 2)
    xx = conv_layer(xx, 256, [1]*3, 7)
    xx = conv_layer(xx, 128, [0.5, 1, 1], 8)
#   
    
    xx = Conv2D(2, 1, padding='same', name='conv9')(xx)
    xx = UpSampling2D(4, name='upsample')(xx)
    
    model = Model(l_in, xx)
    
    model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
    trainer = Model()
    
    allmodel_sum = model.summary()
    
    # Train the model
    history = model.fit(xt, yt, epochs=25, verbose = 1)
    
    output = model.predict(xtest[0].reshape([1, 224, 224, 1])) * 128
    image = np.zeros([224, 224, 3])
    image[:,:,0]=xtest[0][:,:,0]
    image[:,:,1:]=output[0]
    image = color.lab2rgb(image)
    image = img_as_ubyte(image)
    io.imsave("test2.jpg", image)
#    sys.exit()
#    
#    loss = tf.losses.mean_squared_error(labels = ytrue, predictions = xx)
#    cost = tf.reduce_mean(loss)
#    optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(cost)
#    session.run(tf.global_variables_initializer())
#
#    num_epochs = 100
#    for i in range(num_epochs):
#        session.run(optimizer, feed_dict = {x: xt, ytrue:yt})
#        lossvalue = session.run(cost, feed_dict = {x:xt, ytrue : yt})
#        print("epoch: " + str(i) + " loss: " + str(lossvalue))
#
#    print("hej")
#    output = session.run(conv13, feed_dict = {x: xtest[0].reshape([1, 256, 256, 1])})*128
#    image = np.zeros([256, 256, 3])
#    image[:,:,0]=xtest[0][:,:,0]
#    image[:,:,1:]=output[0]
#    image = color.lab2rgb(image)
#    image = img_as_ubyte(image)
#    io.imsave("test.jpg", image)