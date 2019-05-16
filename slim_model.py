import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

# from convNetwork import get_model
import skimage.color as color
import skimage.io as io
import os
import cv2
import numpy as np

'''
RGB Color space: [0,255], [0,255], [0,255] (or [0,1]*3)
YUV Color space: [0,255], [0,255], [0,255] (or [0,1]*3)

RGB White = 255 255 255, Black = 0 0 0
YUV White = 255 128 128, Black = 0 0 0

LAB
The range of the dimensions for RGB and LAB in skimage.color.rgb2lab and lab2rgb are:

rgb_lab:[0,1]x[0,1]x[0,1] -> [0,100] x [-128,128] x [-128,128]
lab_rgb:[0,100] x [-128,128] x [-128,128] --> [0,1]x[0,1]x[0,1]
'''

PIXEL_SIZE = 64
TRAIN_MODEL = True

def upscale2d_layer(inputs, dims):
    return tf.image.resize_nearest_neighbor(inputs, (2*inputs.get_shape().as_list()[1], 2*inputs.get_shape().as_list()[2]))

def get_slim_model(inputs):
    slim = tf.contrib.slim
    padding = 'SAME'
    kernel_size = [3, 3]
    net = slim.conv2d(inputs, 3, kernel_size, padding=padding)
    net = slim.max_pool2d(net, [2,2])
    net = slim.conv2d(net, 8, kernel_size, padding=padding)
    net = slim.max_pool2d(net, [2,2])
    net = slim.conv2d(net, 16, kernel_size, padding=padding)
    net = slim.max_pool2d(net, [2,2])
    net = slim.conv2d(net, 16, kernel_size, padding=padding)
    net = slim.max_pool2d(net, [2,2])
    net = slim.conv2d(net, 32, kernel_size, padding=padding)
    net = slim.max_pool2d(net, [2,2])
    net = slim.conv2d(net, 32, kernel_size, padding=padding)
    # net = slim.max_pool2d(net, [2,2])
    net = slim.conv2d(net, 64, kernel_size, padding=padding)
    # net = upscale2d_layer(net, (PIXEL_SIZE/32, PIXEL_SIZE/32))
    net = slim.conv2d(net, 32, kernel_size, padding=padding)
    net = upscale2d_layer(net, (PIXEL_SIZE/16, PIXEL_SIZE/16))
    net = slim.conv2d(net, 32, kernel_size, padding=padding)
    net = upscale2d_layer(net, (PIXEL_SIZE/8, PIXEL_SIZE/8))
    net = slim.conv2d(net, 16, kernel_size, padding=padding)
    net = upscale2d_layer(net, (PIXEL_SIZE/4, PIXEL_SIZE/4))
    net = slim.conv2d(net, 16, kernel_size, padding=padding)
    net = upscale2d_layer(net, (PIXEL_SIZE/2, PIXEL_SIZE/2))
    net = slim.conv2d(net, 8, kernel_size, padding=padding)
    net = upscale2d_layer(net, (PIXEL_SIZE, PIXEL_SIZE))
    net = slim.conv2d(net, 2, kernel_size, padding=padding)

    # net = slim.conv2d(inputs, 3, kernel_size, padding=padding)
    # batch1 = slim.batch_norm(net)
    #
    # net = slim.conv2d(net, 64, kernel_size, padding=padding)
    # net = slim.conv2d(net, 64, kernel_size, padding=padding)
    # net = slim.max_pool2d(net, [2,2])
    # batch2 = slim.batch_norm(net)
    #
    # net = slim.conv2d(net, 128, kernel_size, padding=padding)
    # net = slim.conv2d(net, 128, kernel_size, padding=padding)
    # net = slim.max_pool2d(net, [2,2])
    # batch3 = slim.batch_norm(net)
    #
    # net = slim.conv2d(net, 256, kernel_size, padding=padding)
    # net = slim.conv2d(net, 256, kernel_size, padding=padding)
    # net = slim.conv2d(net, 256, kernel_size, padding=padding)
    # net = slim.max_pool2d(net, [2,2])
    # batch4 = slim.batch_norm(net)
    #
    # net = slim.conv2d(net, 512, kernel_size, padding=padding)
    # net = slim.conv2d(net, 512, kernel_size, padding=padding)
    # net = slim.conv2d(net, 512, kernel_size, padding=padding)
    # net = slim.max_pool2d(net, [2,2])
    # net = slim.batch_norm(net)

    # Omitted in report
    # net = slim.conv2d(net, 512, kernel_size, padding=padding)
    # net = slim.conv2d(net, 512, kernel_size, padding=padding)
    # net = slim.conv2d(net, 512, kernel_size, padding=padding)

    # net = upscale2d_layer(net, )

    # l_in = Input((224, 224, 1))
    # xx = conv_layer(l_in, 64, [1, 2], 1)
    # xx = conv_layer(xx, 128, [1, 2], 2)
    # xx = conv_layer(xx, 256, [1, 1, 2], 3)
    # xx = conv_layer(xx, 512, [1]*3, 4)
    # xx = conv_layer(xx, 512, [1]*3, 5, 2)
    # xx = conv_layer(xx, 512, [1]*3, 6, 2)
    # xx = conv_layer(xx, 256, [1]*3, 7)
    # xx = conv_layer(xx, 128, [0.5, 1, 1], 8)

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
    # print(image[0])
    image[:,:,:,1:] = output
    # print(image[0])

    image = color.yuv2rgb(image) # returns floats between [0,1]

    for i in range(N):
        # print(image)
        io.imsave(outdir + "/out" + str(i) + ".jpg", image[i], check_contrast=False)


def _reshape_data(x):
    image = x['image']
    image = tf.image.resize(image, (PIXEL_SIZE, PIXEL_SIZE))
    new_im = tf.image.rgb_to_yuv(tf.image.convert_image_dtype(image, tf.float32))
    # print(new_im)
    (image, label) = new_im[:,:,0], new_im[:,:,1:]
    return tf.reshape(image, [PIXEL_SIZE, PIXEL_SIZE, 1]), tf.reshape(label, [PIXEL_SIZE, PIXEL_SIZE, 2])

# tf.enable_eager_execution()
# Get data
data, info = tfds.load("cifar10", with_info=True, split='train')
assert isinstance(data, tf.data.Dataset)
train_dataset = data.map(_reshape_data)
train_dataset = train_dataset.repeat()  # Repeat the input indefinitely.
train_dataset = train_dataset.batch(100)

(test_imgs, labels, N) = get_test_images(with_labels=True)
test_dataset = tf.data.Dataset.from_tensor_slices((test_imgs, labels))
test_dataset = test_dataset.batch(N)
test_dataset = test_dataset.repeat()

iterator = tf.data.Iterator.from_structure(test_dataset.output_types,
                                           test_dataset.output_shapes)

train_init_op = iterator.make_initializer(train_dataset)
test_init_op = iterator.make_initializer(test_dataset)

next_input, next_label = iterator.get_next()

net = get_slim_model(next_input)

loss = tf.losses.mean_squared_error(labels = next_label, predictions = net)
cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    if TRAIN_MODEL:
        sess.run(train_init_op)
        num_epochs = 1000
        for i in range(num_epochs):
            sess.run(optimizer)
            lossvalue = sess.run(cost)
            print("epoch: " + str(i) + " loss: " + str(lossvalue))

        saver.save(sess, "tmp/model.ckpt")
    else:
        saver.restore(sess, "tmp/model.ckpt")

    sess.run(test_init_op)
    res = sess.run(net)

    input_np = sess.run(test_imgs)
    print("INPUT Max: {}, Min: {}, Mean: {}".format(np.max(input_np), np.min(input_np), np.mean(input_np)))
    print("RESULT Max: {}, Min: {}, Mean: {}".format(np.max(res), np.min(res), np.mean(res)))
    # print(res)
    # print(res.shape)
    # print(res)

    save_images_from_prediction(test_imgs, res, N)
