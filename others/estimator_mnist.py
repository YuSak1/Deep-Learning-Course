#!/usr/bin/env python3

#Team: Felipe Vianna and Yuu Sakagushi
# Felipe Vianna: 72ef319b-1ef9-11e8-9de3-00505601122b
# Yuu Sakagushi: d9fbf49b-1c71-11e8-9de3-00505601122b

import numpy as np
import tensorflow as tf

def mnist_model(features, labels, mode, params):
    # TODO: Using features["images"], compute `logits` using:
    # - convolutional layer with 8 channels, kernel size 3 and ReLu activation
    conv1 = tf.layers.conv2d(features["images"], filters=8, kernel_size=[3, 3], activation=tf.nn.relu)
    # - max pooling layer with pool size 2 and stride 2
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)
    # - convolutional layer with 16 channels, kernel size 3 and ReLu activation
    conv2 = tf.layers.conv2d(pool1, filters=16, kernel_size=[3, 3], activation=tf.nn.relu)
    # - max pooling layer with pool size 2 and stride 2
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)
    # - flattening layer
    flatten = tf.layers.flatten(pool2)
    # - dense layer with 256 neurons and ReLU activation
    dense1 = tf.layers.dense(flatten, 256, activation=tf.nn.relu)
    # - dense layer with 10 neurons and no activation
    logits = tf.layers.dense(dense1, 10, activation=None)

    predictions = tf.argmax(logits, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # TODO: Return EstimatorSpec with `mode` and `predictions` parameters
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # TODO: Compute loss using `tf.losses.sparse_softmax_cross_entropy`.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # TODO: Create `eval_metric_ops`, a dictionary with a key "accuracy", its
    # value computed using `tf.metrics.accuracy`.
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions)}

    if mode == tf.estimator.ModeKeys.TRAIN:
        # TODO: Get optimizer class, using `params.get("optimizer", None)`.
        optimizer_class = params.get("optimizer", None)

        # TODO: Create optimizer, using `params.get("learning_rate", None)` parameter.
        optimizer = optimizer_class(learning_rate=params.get("learning_rate", None))

        # TODO: Define `train_op` as `optimizer.minimize`, with `tf.train.get_global_step` as `global_step`.
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        # TODO: Return EstimatorSpec with `mode`, `loss`, `train_op` and `eval_metric_ops` arguments,
        # the latter being the precomputed `eval_metric_ops`.
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)

    if mode == tf.estimator.ModeKeys.EVAL:
        # TODO: Return EstimatorSpec with `mode`, `loss`, `train_op` and `eval_metric_ops`  arguments,
        # the latter being the precomputed `eval_metric_ops`.
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)
    tf.set_random_seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=3, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Construct the model
    model = tf.estimator.Estimator(
        model_fn=mnist_model,
        model_dir=args.logdir,
        config=tf.estimator.RunConfig(tf_random_seed=42,
                                      session_config=tf.ConfigProto(inter_op_parallelism_threads=args.threads,
                                                                    intra_op_parallelism_threads=args.threads)),
        params={
            "optimizer": tf.train.AdamOptimizer,
            "learning_rate": 0.001,
        })

    # Load the data
    from tensorflow.examples.tutorials import mnist
    mnist = mnist.input_data.read_data_sets(".", reshape=False, seed=42)

    # Train
    for i in range(args.epochs):
        # TODO: Define input_fn using `tf.estimator.inputs.numpy_input_fn`.
        # As `x`, pass `{"images": mnist.train images}`, as `y`, pass `mnist.train.labels.astype(np.int64)`,
        # use specified batch_size, one epoch. Normally we would shuffle data with queue capacity 60000,
        # but a random seed cannot be passed to this method; hence, do _not_ shuffle data.
        train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"images": mnist.train.images}, y=mnist.train.labels.astype(np.int64),
                                                            batch_size=args.batch_size, num_epochs=1, shuffle=False)

        # TODO: Train one epoch with `model.train` using the defined input_fn.
        # Note that the `steps` argument should be either left out or set to `None` to respect
        # the `num_epochs` specified when defining `input_fn`.
        model.train(input_fn=train_input_fn, steps=None)


        # TODO: Define validation input_fn similarly, but using `mnist.validation`.
        vali_input_fn = tf.estimator.inputs.numpy_input_fn(x={"images": mnist.validation.images}, y=mnist.validation.labels.astype(np.int64),
                                                            batch_size=args.batch_size, num_epochs=1, shuffle=False)

        # TODO: Evaluate the validation data, using `model.evaluate` with `name="dev"` option
        # and print its return value (which is a dictionary with accuracy, loss and global_step).
        vali_results = model.evaluate(input_fn=vali_input_fn, name="dev")
        print(vali_results)
        
    # TODO: Define input_fn for one epoch of `mnist.test`.
    test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"images": mnist.test.images}, y=mnist.test.labels.astype(np.int64),
                                                            batch_size=args.batch_size, num_epochs=1, shuffle=False)

    # TODO: Evaluate the test set using `model.evaluate` with `name="test"` option
    # and print its return value (which is a dictionary with accuracy, loss and global_step).
    test_results = model.evaluate(input_fn=test_input_fn, name="test")
    print(test_results)
