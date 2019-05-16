import tensorflow as tf

# from convNetwork import get_model
import skimage.color as color
import skimage.io as io
import os
import cv2
import numpy as np

PIXEL_SIZE = 32

def get_slim_model(inputs):
    slim = tf.contrib.slim
    padding = 'SAME'
    kernel_size = [3, 3]
    net = slim.conv2d(inputs, 3, kernel_size, padding=padding)
    # net = slim.max_pool2d(net, [2,2])
    net = slim.conv2d(net, 8, kernel_size, padding=padding)
    # net = slim.max_pool2d(net, [2,2])
    net = slim.conv2d(net, 16, kernel_size, padding=padding)
    # net = slim.max_pool2d(net, [2,2])
    net = slim.conv2d(net, 16, kernel_size, padding=padding)
    # net = slim.max_pool2d(net, [2,2])
    net = slim.conv2d(net, 32, kernel_size, padding=padding)
    # net = slim.max_pool2d(net, [2,2])
    net = slim.conv2d(net, 32, kernel_size, padding=padding)
    # net = slim.max_pool2d(net, [2,2])
    net = slim.conv2d(net, 64, kernel_size, padding=padding)
    net = slim.conv2d(net, 32, kernel_size, padding=padding)
    net = slim.conv2d(net, 32, kernel_size, padding=padding)
    net = slim.conv2d(net, 16, kernel_size, padding=padding)
    net = slim.conv2d(net, 16, kernel_size, padding=padding)
    net = slim.conv2d(net, 8, kernel_size, padding=padding)
    net = slim.conv2d(net, 2, kernel_size, padding=padding)

    return net

def get_test_images(with_labels=False):
    dir = 'inputs'

    # Read input-images for testing
    inputs = [image for image in os.listdir(dir)]
    N = len(inputs)
    inputData = np.zeros([N, PIXEL_SIZE, PIXEL_SIZE, 3])

    for count in range(N):
     img = cv2.resize(io.imread(dir + '/' + inputs[count]), (PIXEL_SIZE, PIXEL_SIZE))
     inputData[count,:,:,:] = img

    # print(inputData[0])
    XInput = tf.image.rgb_to_yuv(tf.image.convert_image_dtype(tf.convert_to_tensor(inputData), tf.float32)) # Needs to be a tensor to run in session
    xI = XInput[:,:,:,0]
    xI = tf.reshape(xI, (N, PIXEL_SIZE, PIXEL_SIZE, 1))
    print(XInput[0])

    if with_labels:
        yI = XInput[:,:,:,1:]
        yI = tf.reshape(yI, (N, PIXEL_SIZE, PIXEL_SIZE, 2))
        return (xI, yI, N)

    return (xI, N)

def save_images_from_prediction(input, output, N):
    outdir = 'outputs'
    image = np.zeros((N, PIXEL_SIZE, PIXEL_SIZE, 3))

    image[:,:,:,0] = input.eval().reshape((N,PIXEL_SIZE,PIXEL_SIZE))
    image[:,:,:,1:] = output

    image = color.yuv2rgb(image) # returns floats between [0,1]

    for i in range(N):
        # print(image)
        io.imsave(outdir + "/out" + str(i) + ".jpg", image[i], check_contrast=False)

(test_imgs, labels, N) = get_test_images(with_labels=True)
image = tf.Variable(tf.zeros((N, PIXEL_SIZE, PIXEL_SIZE, 3)))

net = get_slim_model(test_imgs)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    # saver.save(sess, "tmp/model.ckpt")
    saver.restore(sess, "tmp/model.ckpt")

    res = sess.run(net)*255
    # print(res)
    # print(res.shape)
    # print(res)

    save_images_from_prediction(test_imgs, res, N)
