#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import math

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

    def construct(self, args, train_size):
        with self.session.graph.as_default():
            # TODO: Construct the network and training operation.
             # Inputs
            self.images = tf.placeholder(tf.float32, [None, self.WIDTH, self.HEIGHT, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # Computation
            flattened_images = tf.layers.flatten(self.images, name="flatten")

            conv1 = tf.layers.conv2d(inputs=self.images, filters=10, kernel_size=[5, 5], 
                                                strides=2, padding="same", activation=tf.nn.relu, name="conv1")
            # conv2 = tf.layers.conv2d(inputs=conv1, filters=10, kernel_size=[3, 3], 
            #                                     strides=2, padding="same", activation=tf.nn.relu, name="conv2")
            #pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name="pool1")
            # conv3 = tf.layers.conv2d(inputs=conv2, filters=10, kernel_size=[3, 3], 
            #                                       strides=2, padding="same", activation=tf.nn.relu, name="conv3")
            # conv4 = tf.layers.conv2d(inputs=conv3, filters=10, kernel_size=[3, 3], 
            #                                     strides=2, padding="same", activation=tf.nn.relu, name="conv4")
            

            # flat1 = tf.layers.flatten(inputs=conv1, name = "flat2")
            # dense1 = tf.layers.dense(flat1, 300, activation=tf.nn.sigmoid, name="dense1")

            # pool2 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2, name="pool2")

            #dropout0 = tf.layers.dropout(conv3, rate=0.5, training=self.is_training, name="dropout0")
            flat2 = tf.layers.flatten(inputs=conv1, name = "flat2")
            dense2 = tf.layers.dense(flat2, 500, activation=tf.nn.relu, name="dense2")
            #dense3 = tf.layers.dense(dense2, 200, activation=tf.nn.sigmoid, name="dense3")

            dropout = tf.layers.dropout(dense2, rate=0.5, training=self.is_training, name="dropout")


            output_layer = tf.layers.dense(dropout, self.LABELS, activation=None, name="output_layer")
            self.predictions = tf.argmax(output_layer, axis=1)

            # Training
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")
            global_step = tf.train.create_global_step()
            lr_base = 0.1
            lr_final = 0.01
            
            decay_rate=math.pow((lr_final/lr_base), (1.0/(args.epochs-1)))
            decay_steps = train_size / args.batch_size
            decayed_lr=tf.train.exponential_decay(lr_base, global_step, decay_steps, decay_rate, staircase=True)
            
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.08).minimize(loss, global_step=global_step, name="training")
            self.training = optimizer


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
        return self.session.run([self.summaries[dataset],self.accuracy], {self.images: images, self.labels: labels, self.is_training:False})

    def predict(self, images):
        return self.session.run([self.predictions], {self.images: images, self.is_training:False})



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
    parser.add_argument("--epochs", default=150, type=int, help="Number of epochs.")
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
    mnist = mnist.input_data.read_data_sets("mnist-gan", reshape=False, seed=42,
                                            source_url="https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/mnist-gan/")

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, mnist.train.num_examples)

    # Train
    best = 0
    for i in range(args.epochs):
        if i%10==0:
            print("--------------", i, "/", args.epochs, "epochs")

        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels)

        result = network.evaluate("dev", mnist.validation.images, mnist.validation.labels)
        if result[1] >= best:
            best = result[1]
            print("DEV: {:.2f}".format(result[1]*100), " Best!!!")
        else:
            print("DEV: {:.2f}".format(result[1]*100))
        

    # TODO: Compute test_labels, as numbers 0-9, corresponding to mnist.test.images
    predict = network.predict(mnist.test.images)
    test_labels = predict[0]
    with open("mnist_competition_test.txt", "w") as test_file:
        for label in test_labels:
            test_file.write(str(label)+'\n')

    # BEST: 99.84%
    