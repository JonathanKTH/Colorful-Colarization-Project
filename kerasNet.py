import numpy as np
import cv2
import tensorflow as tf
import skimage.color as color
from skimage.color import rgb2lab, lab2rgb, rgb2gray
import skimage.io as io
from skimage import img_as_ubyte
from skimage.io import imsave
import keras.backend as K
from keras.callbacks import TensorBoard
from keras.models import Model, load_model
from keras.callbacks import LambdaCallback
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Conv2D, Conv2DTranspose, Input, BatchNormalization, UpSampling2D
from keras.optimizers import Adam
from keras.engine.topology import Layer
from keras import Sequential
from keras import losses
from tqdm import tqdm

import matplotlib.pyplot as plt

import os, sys

def convLayer(x, filters, strides=1, idx=1, dilations=1):
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

if __name__ == "__main__":
    DIR_DATA = r'easy_train'
    mydir = r'holder'
    mydirTest = r'images/imgs'
    
    X = []
    for filename in tqdm(os.listdir(DIR_DATA)):
        X.append(img_to_array(load_img(DIR_DATA + '/'+ filename, target_size=(224, 224))))
    X = np.array(X, dtype=float)
    X = 1.0/255*X
    split = int(0.95*len(X))
    Xtrain = X[:split]
    
    imagesTest = [files for files in os.listdir(mydirTest)]

    generator = ImageDataGenerator(shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)
    batch_size = 50
    
    def image_a_b_gen(Xtrain):
       for batch in generator.flow(Xtrain, batch_size=batch_size):
            lab_batch = rgb2lab(batch)
            X_batch = lab_batch[:,:,:,0]
            Y_batch = lab_batch[:,:,:,1:] / 128
            yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)
            
 
    l_in = Input((224, 224, 1))
    xx = convLayer(l_in, 64, [1, 2], 1)
    xx = convLayer(xx, 128, [1, 2], 2)
    xx = convLayer(xx, 256, [1, 1, 2], 3)
    xx = convLayer(xx, 512, [1]*3, 4)
    xx = convLayer(xx, 512, [1]*3, 5, 2)
    xx = convLayer(xx, 512, [1]*3, 6, 2)
    xx = convLayer(xx, 256, [1]*3, 7)
    xx = convLayer(xx, 128, [0.5, 1, 1], 8)
    xx = Conv2D(2, 1, padding='same', name='conv9')(xx)
    xx = UpSampling2D(4, name='upsample')(xx)
    model = Model(l_in, xx)
    model.compile(optimizer='Adam', loss='mse')
    
    model.summary()
    
    sample_count = len(Xtrain)
    batch_size = 32
    steps_per_epoch = sample_count // batch_size
    history = model.fit_generator(image_a_b_gen(Xtrain), epochs=25, steps_per_epoch=steps_per_epoch, verbose=1)
    model.save('Full_model.h5')
    print(history.history.keys())
    
    
    
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('loss_plot.png')
    model = load_model('Full_model.h5')
        
    #Evaluation
    Xtest = rgb2lab(X[split:])[:,:,:,0]
    Xtest = Xtest.reshape(Xtest.shape+(1,))
    Ytest = rgb2lab(X[split:])[:,:,:,1:]
    Ytest = Ytest / 128
    print (model.evaluate(Xtest, Ytest, batch_size=batch_size))
    
    
    # Load greyscaled images
#    prediction_testset = []
#    for filename in os.listdir(mydirTest):
#            prediction_testset.append(img_to_array(load_img(mydirTest + '/' +filename, target_size=(224, 224))))
#    prediction_testset = np.array(prediction_testset, dtype=float)
#    prediction_testset = rgb2lab(1.0/255*prediction_testset)[:,:,:,0]
#    prediction_testset = prediction_testset.reshape(prediction_testset.shape+(1,))
#
#    output = model.predict(prediction_testset)
#    output = output * 128
#    #Save image
#    for i in range(len(output)):
#        img_holder = np.zeros((224, 224, 3))
#        img_holder[:,:,0] = prediction_testset[i][:,:,0]
#        img_holder[:,:,1:] = output[i]
#        img_holder = cv2.resize(lab2rgb(img_holder), (64, 64))
#        imsave("result/img_"+str(i)+".png", img_holder)
#    