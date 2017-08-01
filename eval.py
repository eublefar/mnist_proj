import tensorflow as tf
import mnist

DATA = ['t10k-images.idx3-ubyte',]
LABELS = ['t10k-labels.idx1-ubyte',]
image_batch, label_batch = mnist.build_input_label_batches(DATA, LABELS)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    saver = tf.train.import_meta_graph("checkpoints-11000.meta")

    for i in tf.global_variables():
        print(i.name)
    saver.restore(sess,"checkpoints-11000")
    test = tf.summary.FileWriter("test")
    test.add_graph(sess.graph)

    output = tf.get_collection("output")[0]
    # print(output)
    sess.run(tf.global_variables_initializer())
    # print("outp")
    image_batch_ev, label_batch_ev = sess.run([image_batch, label_batch])
    print("imbatch size = ", image_batch_ev.shape)
    outp = sess.run(output,
    feed_dict={'image_batch_placeholder:0':image_batch_ev.astype(float)})
    print(outp)
    label_batch_ph = tf.placeholder(tf.int32, [15,], name='label_batch_placeholder1')
    argmax = tf.argmax(output, axis = 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(argmax,tf.cast(label_batch_ph, tf.int64)),tf.int64))
    argm = sess.run(argmax,
                        feed_dict={'image_batch_placeholder:0':image_batch_ev,
                                   }
                        )
    accuracy = sess.run(accuracy,
                        feed_dict={'image_batch_placeholder:0':image_batch_ev,
                                   'label_batch_placeholder1:0':label_batch_ev.reshape((15,))
                                   }
                        )
    print("predictions" )
    print(argm)
    print("labels" )
    print(label_batch_ev.reshape((15,)))
    print("accuracy = " + str(accuracy))
    coord.request_stop()
    coord.join(threads)
