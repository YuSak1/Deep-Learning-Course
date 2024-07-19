#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import pandas as pd

# Loads an uppercase dataset.
# - The dataset either uses a specified alphabet, or constructs an alphabet of
#   specified size consisting of most frequent characters.
# - The batches are generated using a sliding window of given size,
#   i.e., for a character, we generate left `window` characters, the character
#   itself and right `window` characters, 2 * `window` +1 in total.
# - The batches can be either generated using `next_batch`+`epoch_finished`,
#   or all data in the original order can be generated using `all_data`.
class Dataset:
    def __init__(self, filename, window, alphabet):
        self._window = window

        # Load the data
        with open(filename, "r", encoding="utf-8") as file:
            self._text = file.read()

        # Create alphabet_map
        self.alphabet_map = {"<pad>": 0, "<unk>": 1}
        if not isinstance(alphabet, int):
            for index, letter in enumerate(alphabet):
                self.alphabet_map[letter] = index
        else:
            # Find most frequent characters
            freqs = {}
            for char in self._text:
                char = char.lower()
                freqs[char] = freqs.get(char, 0) + 1

            most_frequent = sorted(freqs.items(), key=lambda item:item[1], reverse=True)
            for i, (char, freq) in enumerate(most_frequent, len(self.alphabet_map)):
                self.alphabet_map[char] = i
                if len(self.alphabet_map) >= alphabet: break

        #print("Alphabet Map:\n", self.alphabet_map, len(self.alphabet_map))

        # Remap input characters using the alphabet_map
        self._lcletters = np.zeros(len(self._text) + 2 * window, np.uint8)
        self._labels = np.zeros(len(self._text))
        for i in range(len(self._text)):
            char = self._text[i].lower()
            if char not in self.alphabet_map: char = "<unk>"
            # WHAT IS THIS NEXT LINE????
            self._lcletters[i + window] = self.alphabet_map[char]
            self._labels[i] = (1 if self._text[i].isupper() else 0)

        # Compute alphabet
        self._alphabet = [""] * len(self.alphabet_map)
        for key, value in self.alphabet_map.items():
            self._alphabet[value] = key

        #print("Alphabet:\n", self._alphabet)


        self._permutation = np.random.permutation(len(self._text))

    def _create_batch(self, permutation):
        batch_windows = np.zeros([len(permutation), 2 * self._window + 1], np.int32)
        for i in range(0, 2 * self._window + 1):
            batch_windows[:, i] = self._lcletters[permutation + i]
        return batch_windows, self._labels[permutation]

    @property
    def alphabet(self):
        return self._alphabet

    @property
    def text(self):
        return self._text

    @property
    def labels(self):
        return self._labels

    def all_data(self):
        return self._create_batch(np.arange(len(self._text)))

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._create_batch(batch_perm)

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._text))
            return True
        return False


class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, train):
        with self.session.graph.as_default():
            # Inputs
            #self.windows = tf.placeholder(tf.float32, [None, 2 * args.window + 1], name="windows")
            self.labels = tf.placeholder(tf.int64, [None], name="labels") # Or you can use tf.int32
            self.input = tf.placeholder(tf.float32, [args.window,args.alphabet_size], name="input")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # TODO: Define a suitable network with appropriate loss function
            #defining network
            input_layer = tf.layers.flatten(self.input, name="input_layer")
            hidden_layer = tf.layers.dense(input_layer, 300, activation=tf.nn.tanh, name="hidden_layer")
            hidden_layer2 = tf.layers.dense(hidden_layer, 400, activation=tf.nn.tanh, name="hidden_layer2")
            hidden_layer_dropout = tf.layers.dropout(input_layer, rate=0.5, training=self.is_training, name="hidden_layer_dropout")
            output_layer = tf.layers.dense(hidden_layer_dropout, args.window, activation=None, name="output_layer")


            self.predictions = tf.argmax(output_layer, axis=1)

            # TODO: Define training
            # Training
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")
            global_step = tf.train.create_global_step()
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

    def train(self, input, output):
        self.session.run([self.training, self.summaries["train"]], {self.input: input, self.labels: output, self.is_training:True})

    def evaluate(self, dataset, windows, labels):
        return self.session.run(self.summaries[dataset], {self.windows: windows, self.labels: labels, self.is_training:False})

    def recall(self, input):
        return self.session.run([self.predictions], {self.input: input, self.is_training:False})



if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphabet_size", default=60, type=int, help="Alphabet size.")
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--window", default=8, type=int, help="Size of the window to use.")
    parser.add_argument("--hidden_layer", default=30, type=int, help="Size of the hidden layer.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    train = Dataset("uppercase_data_train3.txt", args.window, alphabet=args.alphabet_size)
    dev = Dataset("uppercase_data_dev.txt", args.window, alphabet=train.alphabet)
    test = Dataset("uppercase_data_test.txt", args.window, alphabet=train.alphabet)


    #printing values to understand input and text

    print("lcLtetters:\n",train._lcletters, len(train._lcletters))
    print("Labels:\n",train.labels, len(train._labels))


    #one hot encoding of letters
    list_char = np.zeros(len(train.alphabet_map))
    index=0
    for i in train.alphabet_map:
        list_char[index]=train.alphabet_map[i]
        index+=1

    # encoding alphabet_size characters:
    serie=pd.Series(train.alphabet_map)
    print("pandas serie\n", serie)
    one_hot=pd.get_dummies(serie)
    print(one_hot)
    print("print text :\n",train.text[0:5])

    def encode_text_window(window_word):
        #print("Window: ",window_word)
        one_hot_input=[]
        for i in window_word:
            if i not in train._alphabet:
                one_hot_input = np.append(one_hot_input, one_hot.loc['<unk>'])
            else:
                one_hot_input=np.append(one_hot_input,one_hot.loc[i])

        #print(one_hot_input)
        one_hot_input=np.reshape(one_hot_input, (args.window,args.alphabet_size))
        return one_hot_input.astype(int)


    #checking the labels
    # print("length of text: ",len(train.text))
    # print("length of labels: ",len(train.labels))

    label=train.labels
    labelDict = {}
    for i in label:
        if i not in labelDict:
            labelDict[i]=1
        else:
            labelDict[i]+=1

    #print("LabelDict: \n", labelDict)


    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args,train)

    # Train
    length_text=len(train.text)
    for i in range(args.epochs):
        for j in range(0,length_text - args.window):
            if j%3000==0:
                print(j, "/", len(train.text))
            window_word=train.text[j:j+args.window]
            input=encode_text_window(window_word.lower())
            output=train.labels[j:j+args.window]
            network.train(input,output)


    # test
    f = open('test_result.txt', "w", encoding='UTF-8')
    length_text=len(test.text)
    print("-"*150)
    for i in range(0,length_text - args.window):
        window_word=test.text[i:i+args.window]
        input=encode_text_window(window_word.lower())
        output=train.labels[i:i+args.window]
        out=network.recall(input)
        str = window_word[1]
        out = out[0]
        if out[1] == 1:
            str = str.upper()  
            print("Upper!!!!->",str)
        f.write(str)

        print(str,end="")
    f.close()


    # for i in range(args.epochs):
    #     while not train.epoch_finished():
    #         windows, labels = train.next_batch(args.batch_size)
    #         network.train(windows, labels)
    #
    #     dev_windows, dev_labels = dev.all_data()
    #     network.evaluate("dev", dev_windows, dev_labels)

    # TODO: Generate the uppercased test set
    # f = open('test_result.txt', "w")
    # test_windows, test_labels = test.all_data()
