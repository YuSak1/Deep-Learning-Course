#!/usr/bin/env python3


import numpy as np
import tensorflow as tf

class Network:
    WIDTH = 28
    HEIGHT = 28
    LABELS = 10

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.images = tf.placeholder(tf.float32, [None, self.WIDTH, self.HEIGHT, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # Computation
            # TODO: Add layers described in the args.cnn. Layers are separated by a comma and
            # in addition to the ones allowed in mnist_conv.py you should also support
            # - CB-filters-kernel_size-stride-padding: Add a convolutional layer with BatchNorm
            #   and ReLU activation and specified number of filters, kernel size, stride and padding.
            #   Example: CB-10-3-1-same
            # To correctly implement BatchNorm:
            # - The convolutional layer should not use any activation and no biases.
            # - The output of the convolutional layer is passed to batch_normalization layer, which
            #   should specify `training=True` during training and `training=False` during inference.
            # - The output of the batch_normalization layer is passed through tf.nn.relu.
            # - You need to update the moving averages of mean and variance in the batch normalization
            #   layer during each training batch. Such update operations can be obtained using
            #   `tf.get_collection(tf.GraphKeys.UPDATE_OPS)` and utilized either directly in `session.run`,
            #   or (preferably) attached to `self.train` using `tf.control_dependencies`.
            # Store result in `features`.

            cnn = args.cnn
            cnn = cnn.split(',')
            c = 0
            m = 0
            f = 0
            r = 0
            cb = 0
            features = self.images
            for i in range(len(cnn)):
                cnn_each = cnn[i]
                cnn_each = cnn_each.split('-')
                
                if cnn_each[0] == "C":
                    c += 1
                    c_filters = int(cnn_each[1])
                    c_kernel_size = int(cnn_each[2])
                    c_stride = int(cnn_each[3])
                    c_padding = cnn_each[4]
                    features = tf.layers.conv2d(inputs=features, filters=c_filters, kernel_size=[c_kernel_size, c_kernel_size], 
                                                strides=c_stride, padding=c_padding, activation=tf.nn.relu, name="conv"+str(c))
                    
                elif cnn_each[0] == "CB":
                    cb += 1
                    cb_filters = int(cnn_each[1])
                    cb_kernel_size = int(cnn_each[2])
                    cb_stride = int(cnn_each[3])
                    cb_padding = cnn_each[4]
                    
                    conv = tf.layers.conv2d(inputs=features, filters=cb_filters, kernel_size=[cb_kernel_size, cb_kernel_size], 
                                                strides=cb_stride, padding=cb_padding, use_bias=False, activation=None, name="conv"+str(cb))
                    bn = tf.layers.batch_normalization(inputs=conv, axis=-1,
                                                                  training = self.is_training, name="conv_bn"+str(cb))
                    features = tf.nn.relu(bn)

                elif cnn_each[0] == "M":
                    m += 1
                    m_kernel_size = int(cnn_each[1])
                    m_stride = int(cnn_each[2])
                    features = tf.layers.max_pooling2d(inputs=features, pool_size=[m_kernel_size, m_kernel_size],
                                                    strides=m_stride, name="pool"+str(m))

                elif cnn_each[0] == "F":
                    f += 1
                    features = tf.layers.flatten(inputs=features, name = "flat"+str(f))

                elif cnn_each[0] == "R":
                    r += 1
                    r_hidden_layer_size = int(cnn_each[1])
                    features = tf.layers.dense(inputs=features, units=r_hidden_layer_size, activation=tf.nn.relu, name="hidden_layer"+str(r))


            output_layer = tf.layers.dense(features, self.LABELS, activation=None, name="output_layer")
            self.predictions = tf.argmax(output_layer, axis=1)

            # Training
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")
            global_step = tf.train.create_global_step()
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
              self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, images, labels):
        self.session.run([self.training, self.summaries["train"]], {self.images: images, self.labels: labels, self.is_training:True})

    def evaluate(self, dataset, images, labels):
        accuracy, _ = self.session.run([self.accuracy, self.summaries[dataset]], {self.images: images, self.labels: labels, self.is_training:False})
        return accuracy


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--cnn", default=None, type=str, help="Description of the CNN architecture.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    from tensorflow.examples.tutorials import mnist
    mnist = mnist.input_data.read_data_sets(".", reshape=False, seed=42)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels)

        network.evaluate("dev", mnist.validation.images, mnist.validation.labels)

    accuracy = network.evaluate("test", mnist.test.images, mnist.test.labels)
    print("{:.2f}".format(100 * accuracy))