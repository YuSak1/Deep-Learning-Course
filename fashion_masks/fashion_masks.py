import numpy as np
import tensorflow as tf

class Dataset:
    def __init__(self, filename, shuffle_batches = True):
        data = np.load(filename)
        self._images = data["images"]
        self._labels = data["labels"] if "labels" in data else None
        self._masks = data["masks"] if "masks" in data else None

        self._shuffle_batches = shuffle_batches
        self._permutation = np.random.permutation(len(self._images)) if self._shuffle_batches else range(len(self._images))

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def masks(self):
        return self._masks

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._images[batch_perm], self._labels[batch_perm] if self._labels is not None else None, self._masks[batch_perm] if self._masks is not None else None

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._images)) if self._shuffle_batches else range(len(self._images))
            return True
        return False


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
            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.masks = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="masks")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # TODO: Computation and training.
            #
            # The code below assumes that:
            # - loss is stored in `loss`
            # - training is stored in `self.training`
            # - label predictions are stored in `self.labels_predictions` of shape [None] and type tf.int64
            # - mask predictions are stored in `self.masks_predictions` of shape [None, 28, 28, 1] and type tf.float32
            #   with values 0 or 1

            # Network
            conv1 = tf.layers.conv2d(self.images, filters=15, kernel_size=[5, 5], strides=2, padding="same", activation=tf.nn.sigmoid, name="conv1")
            flatten = tf.layers.flatten(conv1, name="flatten")


            dense1 = tf.layers.dense(flatten, 500, activation=tf.nn.sigmoid, name="dense1")
            dropout1 = tf.layers.dropout(dense1, rate=0.5, training=self.is_training, name="dropout1")
            output_layer1 = tf.layers.dense(dropout1, self.LABELS, activation=None, name="output_layer1")


            dense2 = tf.layers.dense(flatten, 28*28, activation=None, name="dense2")
            output_layer2 = tf.reshape(dense2, [-1, 28, 28, 1])

            self.labels_predictions = tf.argmax(output_layer1, axis=1)
            self.masks_predictions = self.masks_predictions = (tf.sign(output_layer2)+1)*0.5



            # Training
            loss1 = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer1, scope="loss1")
            loss2 = tf.losses.sigmoid_cross_entropy(self.masks, output_layer2, scope="loss2")
            loss = loss1 + loss2
            global_step = tf.train.create_global_step()
            self.training1 = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training1")
            self.training2 = tf.train.AdamOptimizer().minimize(loss2, global_step=global_step, name="training2")


            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.labels_predictions), tf.float32))
            only_correct_masks = tf.where(tf.equal(self.labels, self.labels_predictions),
                                          self.masks_predictions, tf.zeros_like(self.masks_predictions))
            intersection = tf.reduce_sum(only_correct_masks * self.masks, axis=[1,2,3])
            self.iou = tf.reduce_mean(
                intersection / (tf.reduce_sum(only_correct_masks, axis=[1,2,3]) + tf.reduce_sum(self.masks, axis=[1,2,3]) - intersection)
            )

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy),
                                           tf.contrib.summary.scalar("train/iou", self.iou),
                                           tf.contrib.summary.image("train/images", self.images),
                                           tf.contrib.summary.image("train/masks", self.masks_predictions)]
                                           
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset+"/loss", loss),
                                               tf.contrib.summary.scalar(dataset+"/accuracy", self.accuracy),
                                               tf.contrib.summary.scalar(dataset+"/iou", self.iou),
                                               tf.contrib.summary.image(dataset+"/images", self.images),
                                               tf.contrib.summary.image(dataset+"/masks", self.masks_predictions)]
                                               

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, images, labels, masks):
        self.session.run([self.training1, self.summaries["train"]],
                         {self.images: images, self.labels: labels, self.masks: masks, self.is_training: True})
        self.session.run([self.training2, self.summaries["train"]],
                         {self.images: images, self.labels: labels, self.masks: masks, self.is_training: True})

    def evaluate(self, dataset, images, labels, masks):
        return self.session.run([self.summaries[dataset], self.accuracy, self.iou, self.masks_predictions],
                         {self.images: images, self.labels: labels, self.masks: masks, self.is_training: False})

    def predict(self, images):
        return self.session.run([self.labels_predictions, self.masks_predictions],
                                {self.images: images, self.is_training: False})



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
    parser.add_argument("--epochs", default=15, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value)
                  for key, value in sorted(vars(args).items()))).replace("/", "-")
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    train = Dataset("fashion-masks-train.npz")
    dev = Dataset("fashion-masks-dev.npz")
    test = Dataset("fashion-masks-test.npz", shuffle_batches=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    best_acc = 0
    best_mask = 0
    for i in range(args.epochs):
        while not train.epoch_finished():
            images, labels, masks = train.next_batch(args.batch_size)
            network.train(images, labels, masks)


        result = network.evaluate("dev", dev.images, dev.labels, dev.masks)

        if i%5==0:
            print("----------------------", i, "/", args.epochs, "epochs")

        if result[1] >= best_acc:
            best_acc = result[1]
            print("Accuracy: {:.2f}".format(100 * result[1]), 'Best!!!')
        else:
            print("Accuracy: {:.2f}".format(100 * result[1]))
        
        if result[2] >= best_mask:
            best_mask = result[2]
            print("Mask iou: {:.2f}".format(100 * result[2]), 'Best!!!')
        else:
            print("Mask iou: {:.2f}".format(100 * result[2]))
        print("---------------")


    with open("{}/fashion_masks_dev.txt".format(args.logdir), "w", encoding="utf-8") as dev_file:
        while not dev.epoch_finished():
            images, _, _ = dev.next_batch(args.batch_size)
            labels, masks = network.predict(images)
            for i in range(len(labels)):
                print(labels[i], masks[i].astype(np.uint8).flatten(), file=dev_file)

    # Predict test data
    with open("{}/fashion_masks_test.txt".format(args.logdir), "w", encoding="utf-8") as test_file:
        while not test.epoch_finished():
            images, _, _ = test.next_batch(args.batch_size)
            labels, masks = network.predict(images)
            for i in range(len(labels)):
                print(labels[i], *masks[i].astype(np.uint8).flatten(), file=test_file)