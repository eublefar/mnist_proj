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
BATCH_SIZE = 15

reader_data = tf.FixedLengthRecordReader(record_bytes=28*28, header_bytes = 16)
reader_labels = tf.FixedLengthRecordReader(record_bytes=1, header_bytes = 8)

(_,data_rec) = reader_data.read(data_queue)
(_,label_rec) = reader_labels.read(label_queue)

image = tf.decode_raw(data_rec, tf.uint8)
image = tf.reshape(image, [28, 28, 1])
label = tf.decode_raw(label_rec, tf.uint8)
label = tf.reshape(label, [1])


image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                 batch_size=BATCH_SIZE,
                                                 capacity=100,
                                                 min_after_dequeue = 30)






# Convolutional Layer #1 and Pooling Layer #1
conv1 = build_conv_layer(image_batch, CONV[0], True)

 # Convolutional Layer #2 and Pooling Layer #2
conv2 = build_conv_layer(conv1, CONV[1], True)

pool2_flat = tf.reshape(conv2, [BATCH_SIZE, -1])

dense1 = tf.layers.dense(inputs=pool2_flat, units=15, activation=tf.nn.relu)
dropout = tf.layers.dropout(
          inputs=dense1, rate=0.4, training=True)
output = tf.nn.softmax(tf.layers.dense(inputs=dropout, units=10))

onehot_labels = tf.one_hot(indices=tf.cast(tf.reshape(label_batch,[-1]), tf.int32), depth=10)
print(label_batch.get_shape())
print(onehot_labels.get_shape())
loss = tf.losses.softmax_cross_entropy(
  onehot_labels=onehot_labels, logits=output)

train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD")

sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

sess.run(tf.global_variables_initializer())

for value in [label_batch, output, loss]:
    tf.summary.tensor_summary(value.op.name, value)

summaries = tf.summary.merge_all()

summary_writer = tf.summary.FileWriter('log_simple_stats', sess.graph)

image, label, output = sess.run([image_batch[2], label_batch[2], output[2]])
plt.imshow(np.reshape(image, [28,28]));
plt.show()
print("label is " + str(label))
print("output is " + str(output))
print(sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
coord.request_stop()
coord.join(threads)
