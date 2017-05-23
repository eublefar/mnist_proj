import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
CONV = [
        {#first
            'filters':5,
            'kernel_size':[5,5],
            'padding':"same",

            'pool_size':[3,3],
            'strides':2
        },
        {#second
            'filters':10,
            'kernel_size':[3,3],
            'padding':"same",

            'pool_size':[3,3],
            'strides':2
        }
        ]

def build_conv_layer( inp, CONV, pooling):
        conv = tf.layers.conv2d(
          inputs=tf.cast(inp, tf.float32),
          filters=CONV['filters'],
          kernel_size=CONV['kernel_size'],
          padding=CONV['padding'],
          activation=tf.nn.relu)
        if(not pooling):
            return conv
        else:
            pool = tf.layers.max_pooling2d(inputs=conv, pool_size=CONV['pool_size'], strides=CONV['strides'])
            return pool

DATA = 'train-images.idx3-ubyte'
LABELS = 'train-labels.idx1-ubyte'
data_queue = tf.train.string_input_producer([DATA,])
label_queue = tf.train.string_input_producer([LABELS,])
NUM_EPOCHS = 2
BATCH_SIZE = 10
data_string = tf.read_file(DATA)
labels_string = tf.read_file(LABELS)

data_string = tf.substr(data_string, 16 , -1)
pixels = tf.decode_raw(data_string, tf.uint8)
unrolled_images = tf.reshape(pixels, [60000, 28*28, 1])

labels_string = tf.substr(labels_string, 8 , -1)
labels = tf.decode_raw(labels_string, tf.uint8)
labels = tf.reshape(labels, [60000,])
image_batch, label_batch = tf.train.shuffle_batch( [unrolled_images, labels],
                                                     batch_size=BATCH_SIZE,
                                                     capacity=100,
                                                     num_threads=2,
                                                     min_after_dequeue=50)

# # Convolutional Layer #1 and Pooling Layer #1
# conv1 = build_conv_layer(image_batch, CONV[0], True)
#
#  # Convolutional Layer #2 and Pooling Layer #2
# conv2 = build_conv_layer(conv1, CONV[1], True)
#
# pool2_flat = tf.reshape(conv2, [-1, 14999 * 195 * 10])
#
# dense1 = tf.layers.dense(inputs=pool2_flat, units=15, activation=tf.nn.relu)
# dropout = tf.layers.dropout(
#           inputs=dense1, rate=0.4, training=True)
# output = tf.nn.softmax(tf.layers.dense(inputs=dropout, units=10))



sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)


image = image_batch[2][2]
im = image.eval()
print("im_batch shape :" + str(image_batch.get_shape().as_list()))
print("label shape :" + str(label_batch.get_shape().as_list()))
print("label is :" + str(label_batch[2][2].eval()))
# print("output is :" + str(conv1.eval()))

plt.imshow(np.reshape(im, [-1, 28]), cmap='gray')
plt.show()
coord.request_stop()
coord.join(threads)
