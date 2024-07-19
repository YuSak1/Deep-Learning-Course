# coding=utf-8

source_1 = """#!/usr/bin/env python3

#Team: Felipe Vianna and Yuu Sakagushi
# Felipe Vianna: 72ef319b-1ef9-11e8-9de3-00505601122b
# Yuu Sakagushi: d9fbf49b-1c71-11e8-9de3-00505601122b

import numpy as np
import tensorflow as tf
import math

class Dataset:
    def __init__(self, filename, shuffle_batches = True):
        data = np.load(filename)
        self._voxels = data[\"voxels\"]
        self._labels = data[\"labels\"] if \"labels\" in data else None

        self._shuffle_batches = shuffle_batches
        self._new_permutation()

    def _new_permutation(self):
        if self._shuffle_batches:
            self._permutation = np.random.permutation(len(self._voxels))
        else:
            self._permutation = np.arange(len(self._voxels))

    def split(self, ratio):
        split = int(len(self._voxels) * ratio)

        first, second = Dataset.__new__(Dataset), Dataset.__new__(Dataset)
        first._voxels, second._voxels = self._voxels[:split], self._voxels[split:]
        if self._labels is not None:
            first._labels, second._labels = self._labels[:split], self._labels[split:]
        else:
            first._labels, second._labels = None, None

        for dataset in [first, second]:
            dataset._shuffle_batches = self._shuffle_batches
            dataset._new_permutation()

        return first, second

    @property
    def voxels(self):
        return self._voxels

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._voxels[batch_perm], self._labels[batch_perm] if self._labels is not None else None

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._new_permutation()
            return True
        return False


class Network:
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
            self.voxels = tf.placeholder(tf.float32, [None, args.modelnet_dim, args.modelnet_dim, args.modelnet_dim, 1], name=\"voxels\")
            self.labels = tf.placeholder(tf.int64, [None], name=\"labels\")
            self.is_training = tf.placeholder(tf.bool, [], name=\"is_training\")

            # TODO: Computation and training.
            #
            # The code below assumes that:
            # - loss is stored in `loss`
            # - training is stored in `self.training`
            # - label predictions are stored in `self.predictions`
            #flatten = tf.layers.flatten(self.voxels, name=\"flatten\")
            conv1 = tf.layers.conv3d(inputs=self.voxels, filters=15, kernel_size=[3, 3, 3], strides=1, padding=\"same\", activation=tf.nn.relu)
            conv2 = tf.layers.conv3d(inputs=conv1, filters=10, kernel_size=[3, 3, 3], strides=1, padding=\"same\", activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=2)
            conv3 = tf.layers.conv3d(inputs=pool1, filters=10, kernel_size=[3, 3, 3], strides=1, padding=\"same\", activation=tf.nn.relu)
            
            # pool2 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2, name=\"pool2\")

            # dropout0 = tf.layers.dropout(conv3, rate=0.5, training=self.is_training)
            flat1 = tf.layers.flatten(conv3)
            dense1 = tf.layers.dense(flat1, 1200, activation=tf.nn.relu)
            #dense3 = tf.layers.dense(dense2, 200, activation=tf.nn.sigmoid)

            dropout = tf.layers.dropout(dense1, rate=0.5, training=self.is_training)
            output_layer = tf.layers.dense(dropout, self.LABELS, activation=None)
            self.predictions = tf.argmax(output_layer, axis=1)

            # Training
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope=\"loss\")
            global_step = tf.train.create_global_step()
            # lr_base = 0.1
            # lr_final = 0.01
            
            # decay_rate=math.pow((lr_final/lr_base), (1.0/(args.epochs-1)))
            # decay_steps = train_size / args.batch_size
            # decayed_lr=tf.train.exponential_decay(lr_base, global_step, decay_steps, decay_rate, staircase=True)
            
            optimizer = tf.train.AdamOptimizer(learning_rate=0.005,).minimize(loss, global_step=global_step, name=\"training\")
            self.training = optimizer


            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(8):
                self.summaries[\"train\"] = [tf.contrib.summary.scalar(\"train/loss\", loss),
                                           tf.contrib.summary.scalar(\"train/accuracy\", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in [\"dev\", \"test\"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + \"/loss\", loss),
                                               tf.contrib.summary.scalar(dataset + \"/accuracy\", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, voxels, labels):
        self.session.run([self.training, self.summaries[\"train\"]], {self.voxels: voxels, self.labels: labels, self.is_training: True})

    def evaluate(self, dataset, voxels, labels):
        accuracy, _ = self.session.run([self.accuracy, self.summaries[dataset]], {self.voxels: voxels, self.labels: labels, self.is_training: False})
        return accuracy

    def predict(self, voxels):
        return self.session.run(self.predictions, {self.voxels: voxels, self.is_training: False})


if __name__ == \"__main__\":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--batch_size\", default=50, type=int, help=\"Batch size.\")
    parser.add_argument(\"--epochs\", default=15, type=int, help=\"Number of epochs.\")
    parser.add_argument(\"--modelnet_dim\", default=20, type=int, help=\"Dimension of ModelNet data.\")
    parser.add_argument(\"--threads\", default=1, type=int, help=\"Maximum number of threads to use.\")
    parser.add_argument(\"--train_split\", default=0.9, type=float, help=\"Ratio of examples to use as train.\")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = \"logs/{}-{}-{}\".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime(\"%Y-%m-%d_%H%M%S\"),
        \",\".join((\"{}={}\".format(re.sub(\"(.)[^_]*_?\", r\"\\1\", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists(\"logs\"): os.mkdir(\"logs\") # TF 1.6 will do this by itself

    # Load the data
    train, dev = Dataset(\"modelnet{}-train.npz\".format(args.modelnet_dim)).split(args.train_split)
    test = Dataset(\"modelnet{}-test.npz\".format(args.modelnet_dim), shuffle_batches=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        while not train.epoch_finished():
            voxels, labels = train.next_batch(args.batch_size)
            network.train(voxels, labels)

        accuracy = network.evaluate(\"dev\", dev.voxels, dev.labels)
        print(\"{:.2f}\".format(100 * accuracy))

    # Predict test data
    with open(\"3d_recognition_test.txt\", \"w\") as test_file:
        while not test.epoch_finished():
            voxels, _ = test.next_batch(args.batch_size)
            labels = network.predict(voxels)

            for label in labels:
                print(label, file=test_file)

    # best: 91.99%"""

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;0mJxiCq8~gFQ{NIp$$7Aq+YKHC7vzcC6zX1&#z%cj3+{m2tbPH=Y{l55;$K&vuYwZt%{5`vmuBmUjYJRF^xaiHd3yI>oxfL}Ii&U532a=7&v_j*hH$L^#PWmHZ5~3d=u#GP$1-k_grvzfryoov~|GYnmpsFW6P62Q`1qY*v1iD=&l69c@<*zpi5&O+JD0;?c3bKByuDGvV<{glQXe7Ljw!e>VslNn9em)H~+M?(s4O_=Dt+#k%;zNLV4=c~e+bQMk%Qk`vWJ$(e6gCpcIn&5(3@aT^Qqkf=Fy`vglS4f`p=|5XcFHaV>$1qAP!LoIhdHyifhIP8u;am~54Mbq3uSd3Pp-a%a*G|g>R)MH!6^E;mFn^j{VfP-l~<%`TG#)U9dk1Ygmg&e(eK25)9`mwt6TWNf~zrm4b=$d2@jr9KQ8p3=(AIl`#e?b4qFnA&+Z0@RXVoI~#nDhVPV3%9uX6gz;SGF@OZ}(tQ`D?Vzm~6SR`H>iRyK)SvXOtG9_{{Fsc!N&@P*&}OH^w?^h$Ille5nWf8EGf48EvQHa}QPhb3jC1lL8fjI@r~m%2`m4Yau_>=4yJ5=NXDJO$ln*55akO>LsfQWCo+Rz86}vKuv^WrLCNG^g&31o8UMsk1G9Ynx$<dmo^epXjYZ|S#*PFnt3bS!nreYo+JgJ5g;@aaQ2w&sgdE~LC|uiG8<#l64}BDl>AyTUxhl&zq8v-U?Jp1KV8QE#DFi{!ss9#VjSg+0+3lVkPWb<LHIJU)izw!=Jri*jcMq)Pv;S0U}Ksz7{scvEb0fgD(ZH{HI(%(@duKX2a?=XbwgMg?jFuR00000)+jkORmdMc00E^1q!j=FIW&NpvBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys
    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())
