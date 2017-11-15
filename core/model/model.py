import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

from utils import Utils


class RNN:

    def __init__(self, layers, n_classes, eta=0.01):

        self.layers = layers
        self.eta = eta
        self.n_classes = n_classes
        self.dropout_prob = tf.placeholder(tf.float32, name='pkeep')

    @classmethod
    def from_args(cls, args, n_classes):

        new_rnn = RNN(args.layers, n_classes, args.eta,)

        return new_rnn

    def get_states(self):

        return self.initial_state, self.last_state

    def create_variables(self):

        self.x = tf.placeholder(tf.int64, [None, None], name='X')
        self.y = tf.placeholder(tf.int64, [None, None], name='Y')
        # each input/target must be a 1-hot vector
        self.x = tf.one_hot(self.x, depth=self.n_classes, name='X_hot')
        self.y = tf.one_hot(self.y, depth=self.n_classes, name='Y_hot')

    def create_rnn_layers(self, cell=rnn.LSTMCell):

        cells = [cell(size) for size in self.layers]

        cells = [rnn.DropoutWrapper(cell,input_keep_prob=self.dropout_prob, output_keep_prob=self.dropout_prob) for cell in cells]

        cells = rnn.MultiRNNCell(cells, state_is_tuple=True)

        return cells

    def run_rnn(self):

        rnn_layers = self.rnn_layers = self.create_rnn_layers()

        self.initial_state = rnn_layers.zero_state(tf.shape(self.x)[0], dtype=tf.float32)

        output, self.last_state = tf.nn.dynamic_rnn(rnn_layers, self.x, initial_state=self.initial_state ,dtype=tf.float32)


        return output

    def run_linear(self, rnn_output):

        shape = self.x.get_shape().as_list()

        rnn_output_flat = tf.reshape(rnn_output, [-1, self.layers[-1]])

        W = tf.Variable(tf.truncated_normal([self.layers[-1], shape[-1]], stddev=0.1), name='W')
        b = tf.Variable(tf.zeros([shape[-1]]), name='b')

        output_linear = tf.matmul(rnn_output_flat, W) + b

        output_linear_activated = tf.nn.softmax(output_linear)

        return output_linear, output_linear_activated


    def train_step(self, loss):

        train_step = tf.train.AdamOptimizer(self.eta).minimize(loss)

        return train_step


    def get_loss(self, logits):

        shape = self.x.get_shape().as_list()

        y_flat = tf.reshape(self.y, [-1, shape[-1]])

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_flat)

        return loss

    def get_pred(self, output):

        pred = tf.argmax(output, 1)

        pred = tf.reshape(pred, [tf.shape(self.x)[0], - 1])

        return pred

    def get_accuracy(self, logits):

        shape = self.x.get_shape().as_list()

        y_flat = tf.reshape(self.y, [-1, shape[-1]])

        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y_flat, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        return accuracy

    def build(self):
        """
        Create the full graph for this model
        :return: The tf computation nodes
        """

        self.create_variables()
        # create model
        rnn_output = self.run_rnn()
        output_linear, output_linear_activated = self.run_linear(rnn_output)
        # save all predictions
        self.preds = preds = output_linear_activated
        # store the correct one
        self.pred = pred = self.get_pred(output_linear_activated)

        loss = self.get_loss(output_linear)

        train_step = self.train_step(loss)

        accuracy = self.get_accuracy(output_linear)

        cost = tf.reduce_mean(loss)

        return pred, preds, cost, train_step, accuracy,

    def generate(self, input_val, sess, interactive=False, n_text=100):
        """
        Genere text from a input
        :param input_val: The initial start text
        :param sess: The current tf session
        :param interactive: If True, it will print while generating
        :param n_text: How much text we have to generate
        :return: The generated text
        """
        text = input_val
        x_batch = np.array([[ord(c) for c in input_val]])

        last_state = None

        if (interactive):
            print(input_val,end='')

        for i in range(n_text):
            feed_dict = {'X:0': x_batch, 'pkeep:0':1.0}

            if (last_state != None):
                feed_dict[self.initial_state] = last_state

            preds, last_state = sess.run([self.preds, self.last_state], feed_dict=feed_dict)
            preds = np.array([preds[-1]])

            next = Utils.sample_prob_picker_from_best(preds)

            if(interactive):
                print(chr(next[0]),end='')
            # next is 1D vector like [14]
            text += "".join(chr(next[0]))

            x_batch = np.array([next])

        return text

