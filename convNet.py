import numpy as np
import cv2
import tensorflow as tf
import skimage.color as color
from skimage.color import rgb2lab, lab2rgb, rgb2gray

import skimage.io as io
from skimage import img_as_ubyte
from skimage.io import imsave

#from keras.datasets import cifar10

#NEW
import keras.backend as K
from keras.callbacks import TensorBoard

#import tensorflow_datasets as tfds

from keras.models import Model, load_model
from keras.callbacks import LambdaCallback
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Conv2D, Conv2DTranspose, Input, BatchNormalization, UpSampling2D
from keras.optimizers import Adam
from keras.engine.topology import Layer
from keras import Sequential
from keras import losses
#from colorizer import lab_to_rgb, rgb_to_lab
from tqdm import tqdm



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

#class LAB(Layer):
#    def call(self, x):
#        l, ab_truth = rgb_to_lab(x / 255)
#        return [l, ab_truth]
#    
#    def compute_output_shape(self, input_shape):
#        input_shape = np.array(input_shape)
#        l_shape = input_shape.copy()
#        ab_shape = input_shape.copy()
#        l_shape[-1] = 1
#        ab_shape[-1] = 2
#        return [tuple(l_shape), tuple(ab_shape)]
#    
#class MeanSquaredError(Layer):
#    def call(self, x):
#        ab_true, ab_pred = x
#        return losses.mean_squared_error(ab_pred, ab_truth)
#    
#    def compute_output_shape(self, input_shape):
#        return (None, 1)    
#

#def identity_error(dummy_target, loss):
#    return K.mean(loss, axis=-1)

#def image_a_b_gen(Xtrain, batches):
#   for batch in generator.flow(Xtrain, batch_size=batch_size):
#        lab_batch = rgb2lab(batch)
#        X_batch = lab_batch[:,:,:,0]
#        Y_batch = lab_batch[:,:,:,1:] / 128
#        yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)
#        
if __name__ == "__main__":
    DIR_DATA = r'images/val_images'
    mydir = r'holder'
    mydirTest = r'imgsTest'
    images = [files for files in os.listdir(mydir)]
    
    X = []
    for filename in tqdm(os.listdir(DIR_DATA)):
        #X.append(img_to_array(cv2.resize(load_img('images/imgs/'+filename), (224, 224))))
        #img = cv2.resize(io.imread(DIR_DATA + '/'+ filename), (224, 224))
        X.append(img_to_array(load_img(DIR_DATA + '/'+ filename, target_size=(224, 224))))
    X = np.array(X, dtype=float)
    X = 1.0/255*X
    split = int(0.95*len(X))
    Xtrain = X[:split]
    
    
    imagesTest = [files for files in os.listdir(mydirTest)]
    #print("Hej")
    
    ## ATTEMPT TO import batches. 
    generator = ImageDataGenerator(shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)
    batch_size = 50
    
    def image_a_b_gen(Xtrain, batches):
       for batch in generator.flow(Xtrain, batch_size=batch_size):
            lab_batch = rgb2lab(batch)
            X_batch = lab_batch[:,:,:,0]
            Y_batch = lab_batch[:,:,:,1:] / 128
            yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)
            
 
    l_in = Input((224, 224, 1))
    xx = conv_layer(l_in, 64, [1, 2], 1)
    xx = conv_layer(xx, 128, [1, 2], 2)
    xx = conv_layer(xx, 256, [1, 1, 2], 3)
    xx = conv_layer(xx, 512, [1]*3, 4)
    xx = conv_layer(xx, 512, [1]*3, 5, 2)
    xx = conv_layer(xx, 512, [1]*3, 6, 2)
    xx = conv_layer(xx, 256, [1]*3, 7)
    xx = conv_layer(xx, 128, [0.5, 1, 1], 8)
   
    
    xx = Conv2D(2, 1, padding='same', name='conv9')(xx)
    xx = UpSampling2D(4, name='upsample')(xx)
    
    model = Model(l_in, xx)
    model.summary()
    
    model.compile(optimizer='Adam', loss='mse')
    tensorboard = TensorBoard(log_dir='Graph', histogram_freq=1,  
          write_graph=True, write_images=True)
    tensorboard.set_model(model)
    sample_count = len(Xtrain)
    batch_size = 32
    steps_per_epoch = sample_count // batch_size
    #steps = 
    model.fit_generator(image_a_b_gen(Xtrain, batch_size), epochs=50, steps_per_epoch=steps_per_epoch, verbose=1, callbacks=[tensorboard])
    model.save(f'Full_model.h5')
    
    
    #model = load_model('model2000.h5')
    
    
    #TEST MODEL
    Xtest = rgb2lab(1.0/255*X[split:])[:,:,:,0]
    Xtest = Xtest.reshape(Xtest.shape+(1,))
    Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]
    Ytest = Ytest / 128
    print (model.evaluate(Xtest, Ytest, batch_size=batch_size))
    # Load black and white images
#    color_me = []
#    for filename in os.listdir(mydirTest):
#            color_me.append(img_to_array(load_img(mydirTest + '/' +filename, target_size=(224, 224))))
#    color_me = np.array(color_me, dtype=float)
#    color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
#    color_me = color_me.reshape(color_me.shape+(1,))
#    # Test model
#    output = model.predict(color_me)
#    output = output * 128
#    # Output colorizations
#    for i in range(len(output)):
#            cur = np.zeros((224, 224, 3))
#            cur[:,:,0] = color_me[i][:,:,0]
#            cur[:,:,1:] = output[i]
#            imsave("result/img_"+str(i)+".png", lab2rgb(cur))
    # DEFINE MODEL, where we greyscale images as well. 
#    img_input = Input((None, None, 3), name='img_input')
#    
#    l, ab_truth = LAB(name='lab')(img_input)
#    ab_pred = model(l)
    
    #loss = losses.mean_quared_error(ab_pred, ab_truth)
    #loss = MeanSquaredError(name='mean_squared_error')([ab_truth, ab_pred])
    
#    trainer = Model(img_input, ab_pred)
#    trainer.compile(Adam(3e-5, beta_2=0.99, decay=1e-3), identity_error)
#    #trainer.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#    
#    trainer.summary()
#    
#    
#    
##    model.compile(optimizer='adam',
##              loss='mean_squared_error',
##              metrics=['accuracy'])
##    
#    
#    # Train the model
#    #history = model.fit(xt, yt, epochs=25, verbose = 1)
#    
#    epochs = 10
#    
#    for epoch in tqdm(range(epochs)):
#        
#        hist = trainer.fit_generator(train_batches, train_batches.batches_per_epoch,
#                    verbose=0)
#        print(hist.history['loss'])
#    print("done")
#    
#    #Load test data as tensor, then use this function. Use 
#    l, ab_truth = LAB(name='lab')(xtest[0])
#    
    
#    output = model.predict(xtest[0].reshape([1, 224, 224, 1]))
#    
#    image = np.zeros([224, 224, 3])
#    image[:,:,0]=xtest[0][:,:,0]
#    image[:,:,1:]=output[0]
#    image = color.lab2rgb(image)
#    image = img_as_ubyte(image)
#    io.imsave("test4.jpg", image)
    
#    output = model.predict(xtest[0].reshape([1, 224, 224, 1])) * 128
#    image = np.zeros([224, 224, 3])
#    image[:,:,0]=xtest[0][:,:,0]
#    image[:,:,1:]=output[0]
#    image = color.lab2rgb(image)
#    image = img_as_ubyte(image)
#    io.imsave("test2.jpg", image)
#    
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