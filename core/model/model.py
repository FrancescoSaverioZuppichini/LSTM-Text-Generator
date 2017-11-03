import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class RNN:

    def build(self,x, y, layers, n_classes, eta, dropout):

        w = tf.Variable(tf.truncated_norma([layers[-1], n_classes]))
        b = tf.Variable(tf.truncated_norma([n_classes]))

        if(dropout):
            rnn_cell = rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.BasicLSTMCell(n_size, state_is_tuple=True), 0.5) for n_size in layers], state_is_tuple=True)
        else:
            rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_size) for n_size in layers], state_is_tuple=True)

        initial_state = rnn_cell.zero_state(tf.shape(x)[0], dtype=tf.float32)

        outputs, state = tf.nn.dynamic_rnn(rnn_cell, x, initial_state=initial_state ,dtype=tf.float32)
        # reshape by the last layer dimension
        pack_pred = tf.reshape(outputs, [-1 , layers[-1]])

        pred = tf.matmul(pack_pred, w) + b

        flat_Y = tf.reshape(y, [-1 , n_classes])

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=flat_Y))

        train_step = tf.train.AdamOptimizer(eta).minimize(cost)

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(flat_Y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.nodes = {
            'pred': pred,
            'cost': cost,
            'train': train_step,
            'accuracy': accuracy,
            'correct_pred': correct_pred
        }

        return pred, cost, train_step, accuracy, correct_pred

    def generate(self, x, y, input_val, sess, n_classes, r, n_text=100):
        text = ''
        x_batch = np.array([r.encode_array(list(input_val))])

        x_hot = tf.one_hot(x_batch, depth=n_classes, on_value=1.0)

        for i in range(n_text):
            preds = sess.run(self.nodes['pred'], feed_dict={x: x_hot.eval()})
            keys = np.argmax(preds, axis=1)

            text += ''.join(r.decode_array(keys))

            preds = keys.reshape([len(x_batch), len(x_batch[0])])
            x_hot = tf.one_hot(preds, depth=n_classes, on_value=1.0)

        open('output.txt', 'w').write(text)
        print(text)