import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers as l

import numpy as np

class RNN:

    def __init__(self, x, y, layers, eta=0.01, dropout=False):

        self.x = x
        self.y = y
        self.layers = layers
        self.eta = eta
        self.dropout = dropout
        self.initial_state = None
        self.last_state = None

    @classmethod
    def from_args(cls,x,y, args):

        new_rnn = RNN(x,y,args.layers, args.eta, args.dropout)

        return new_rnn

    def get_states(self):

        return self.initial_state, self.last_state

    def create_rnn_layers(self, cell=rnn.BasicLSTMCell):

        cells = [cell(size) for size in self.layers]

        if(self.dropout):
            cells = [rnn.DropoutWrapper(cell,input_keep_prob=0.8) for cell in cells]

        cells = rnn.MultiRNNCell(cells, state_is_tuple=True)

        if(self.dropout):
            cells = rnn.DropoutWrapper(cells, output_keep_prob=0.8)

        return cells

    def run_rnn(self):
        rnn_layers = self.create_rnn_layers()

        self.initial_state = rnn_layers.zero_state(tf.shape(self.x)[0], dtype=tf.float32)

        output, self.last_state = tf.nn.dynamic_rnn(rnn_layers, self.x, initial_state=self.initial_state ,dtype=tf.float32)


        return output

    def run_linear(self, rnn_output):
        shape = self.x.get_shape().as_list()

        rnn_output_flat = tf.reshape(rnn_output, [-1, self.layers[-1]])

        output_linear = l.linear(rnn_output_flat, shape[-1])

        output_linear_activated = tf.nn.softmax(output_linear)

        return output_linear, output_linear_activated


    def train_step(self, loss):

        train_step = tf.train.AdamOptimizer(self.eta).minimize(loss)

        return train_step


    def get_loss(self, logits):
        shape = self.x.get_shape().as_list()

        y_flat = tf.reshape(self.y, [-1, shape[-1]])  # [BATCH_SIZE x LEN_SEQ, SYMBOLS]

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_flat)

        return loss

    def get_pred(self, output):

        pred = tf.argmax(output, 1)
        pred = tf.reshape(pred, [tf.shape(self.x)[0], - 1])

        return pred

    def get_accuracy(self, logits):
        shape = self.x.get_shape().as_list()

        y_flat = tf.reshape(self.y, [-1, shape[-1]])  # [BATCH_SIZE x LEN_SEQ, SYMBOLS]

        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y_flat, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        return accuracy

    def build(self,x, y, layers, n_classes, eta, dropout, state=None, temperature=1.0):

        rnn_output = self.run_rnn()
        output_linear, output_linear_activated = self.run_linear(rnn_output)

        self.pred = pred = self.get_pred(output_linear_activated)

        loss = self.get_loss(output_linear)

        train_step = self.train_step(loss)

        accuracy = self.get_accuracy(output_linear, )

        cost = tf.reduce_mean(loss)

        return pred, cost, train_step, accuracy

    def generate(self, x, y, input_val, sess, n_classes, r, n_text=100):
        text = input_val
        x_batch = np.array([r.encode_array(list(input_val))])

        x_hot = tf.one_hot(x_batch, depth=n_classes, on_value=1.0)

        for i in range(n_text):

            last_state = sess.run(self.last_state, feed_dict={x: x_hot.eval()})

            preds = sess.run(self.pred, feed_dict={x: x_hot.eval(), self.initial_state: last_state})
            # keys = np.argmax(preds, axis=1)
            keys = preds
            text += "".join(r.decode_array(keys[0]))

            preds = keys.reshape([len(x_batch), len(x_batch[0])])
            x_hot = tf.one_hot(preds, depth=n_classes, on_value=1.0)

        return text

