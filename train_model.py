import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

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

def _reshape_data(x):
    image = x['image']
    new_im = tf.image.rgb_to_yuv(tf.image.convert_image_dtype(image, tf.float32))
    # print(new_im)
    (image, label) = new_im[:,:,0], new_im[:,:,1:]
    return tf.reshape(image, [PIXEL_SIZE, PIXEL_SIZE, 1]), tf.reshape(label, [PIXEL_SIZE, PIXEL_SIZE, 2])

# tf.enable_eager_execution()
# Get data
data, info = tfds.load("cifar10", with_info=True, split='train')
assert isinstance(data, tf.data.Dataset)
dataset = data.map(_reshape_data)
dataset = dataset.repeat()  # Repeat the input indefinitely.
dataset = dataset.batch(1000)

iterator = dataset.make_one_shot_iterator()
next_input, next_label = iterator.get_next()

net = get_slim_model(next_input)

loss = tf.losses.mean_squared_error(labels = next_label, predictions = net)
cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(cost)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    num_epochs = 1000
    for i in range(num_epochs):
        sess.run(optimizer)
        lossvalue = sess.run(cost)
        print("epoch: " + str(i) + " loss: " + str(lossvalue))

    saver = tf.train.Saver()
    saver.save(sess, "tmp/model.ckpt")
