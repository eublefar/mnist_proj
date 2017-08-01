import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import threading
import sys
save_and_exit = False

CONV = [
        {#first
            'filters':24,
            'kernel_size':[5,5],
            'padding':"same",

            'pool_size':[3,3],
            'strides':2
        },
        {#second
            'filters':20,
            'kernel_size':[3,3],
            'padding':"same",

            'pool_size':[3,3],
            'strides':2
        }
        ]
def build_forward_prop(image_batch, params, batch_size = 15):
    # Convolutional Layer #1 and Pooling Layer #1
    conv1 = build_conv_layer(image_batch, params[0], False)

     # Convolutional Layer #2 and Pooling Layer #2
    conv2 = build_conv_layer(conv1, params[1], False)

    pool2_flat = tf.reshape(conv2, [batch_size, -1])

    dense1 = tf.layers.dense(inputs=pool2_flat, units=30, activation=tf.nn.relu)
    output = tf.nn.softmax(tf.layers.dense(inputs=dense1, units=10))
    return output

def build_loss(output, label_batch):
    onehot_labels = tf.one_hot(indices=tf.cast(tf.reshape(label_batch,[-1]), tf.int32), depth=10)
    print(label_batch.get_shape())
    print(onehot_labels.get_shape())
    loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=output)
    return loss

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

def build_input_label_batches(input_filename_queue, label_filename_queue, batch_size = 15, epochs = 2):
    data_queue = tf.train.string_input_producer(input_filename_queue)
    label_queue = tf.train.string_input_producer(label_filename_queue)
    reader_data = tf.FixedLengthRecordReader(record_bytes=28*28, header_bytes = 16)
    reader_labels = tf.FixedLengthRecordReader(record_bytes=1, header_bytes = 8)
    (_,data_rec) = reader_data.read(data_queue)
    (_,label_rec) = reader_labels.read(label_queue)
    image = tf.decode_raw(data_rec, tf.uint8)
    image = tf.reshape(image, [28, 28, 1])
    label = tf.decode_raw(label_rec, tf.uint8)
    label = tf.reshape(label, [1])
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                     batch_size=batch_size,
                                                     capacity=100,
                                                     min_after_dequeue = 30)
    return [image_batch, label_batch]




if __name__ == "__main__":
    DATA = ['train-images.idx3-ubyte',]
    LABELS = ['train-labels.idx1-ubyte',]

    image_batch, label_batch = build_input_label_batches(DATA, LABELS)

    image_batch_placeholder = tf.placeholder(tf.float32, image_batch.get_shape(), name = 'image_batch_placeholder')
    label_batch_placeholder = tf.placeholder(tf.float32, label_batch.get_shape(), name = 'label_batch_placeholder')

    output = build_forward_prop(image_batch_placeholder, CONV)

    loss = build_loss(output, label_batch_placeholder)

    global_step = tf.Variable(0,name='global_step',trainable=False)
    train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss, global_step = global_step)

    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv2d'):
        tf.summary.histogram(var.name, var)
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv2d_1'):
        tf.summary.histogram(var.name, var)
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dense'):
        tf.summary.histogram(var.name, var)
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dense_1'):
        tf.summary.histogram(var.name, var)
    tf.summary.scalar("loss", loss)

    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    sess.run(tf.global_variables_initializer())

    summaries = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter('log_simple_stats', sess.graph)
    tf.add_to_collection('output', output)
    tf.add_to_collection('loss', loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    for i in tf.global_variables():
        print(i.name)

    for i in range(240000):
        image_batch_ev, label_batch_ev = sess.run([image_batch, label_batch])
        _, merged = sess.run([train_op, summaries],
                             feed_dict={'image_batch_placeholder:0':image_batch_ev,
                                        'label_batch_placeholder:0':label_batch_ev})
        summary_writer.add_summary(merged, i)
        if i%1000 == 0 :
            saver.save(sess, save_path="./checkpoints", global_step=i)


    # image, label, output = sess.run([image_batch[2], label_batch[2], output[2]])
    plt.imshow(np.reshape(image, [28,28]));
    plt.show()
    print("label is " + str(label))
    print("output is " + str(output))
    print(sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    coord.request_stop()
    coord.join(threads)
