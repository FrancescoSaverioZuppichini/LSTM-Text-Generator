import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
import numpy as np

class RNN:

    def build(self,x, y, layers, n_classes, eta, dropout):
        shape = x.get_shape().as_list()

        w = tf.Variable(tf.truncated_normal([layers[-1], shape[-1]]))
        b = tf.Variable(tf.truncated_normal([shape[-1]]))

        cells = [rnn.BasicLSTMCell(n_size) for n_size in layers]

        if(dropout):
            cells = [rnn.DropoutWrapper(cell,0.8) for cell in cells]

        rnn_cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

        initial_state = rnn_cell.zero_state(tf.shape(x)[0], dtype=tf.float32)

        outputs, state = tf.nn.dynamic_rnn(rnn_cell, x, initial_state=initial_state ,dtype=tf.float32)
        # reshape by the last layer dimension
        y_flat = tf.reshape(outputs, [-1 , layers[-1]])

        y_logits = tf.matmul(y_flat, w) + b

        y_flat = tf.reshape(y, [-1 , shape[-1]])

        cost = tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y_flat)
        cost = tf.reduce_mean(cost)

        train_step = tf.train.AdamOptimizer(eta).minimize(cost)

        correct_pred = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y_flat, 1))


        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        predictions = tf.nn.softmax(y_logits)

        self.nodes = {
            'pred': predictions,
            'cost': cost,
            'train': train_step,
            'accuracy': accuracy,
            'correct_pred': correct_pred
        }

        return predictions, cost, train_step, accuracy, correct_pred

    def generate(self, x, y, input_val, sess, n_classes, r, n_text=100):
        text = input_val
        x_batch = np.array([r.encode_array(list(input_val))])

        x_hot = tf.one_hot(x_batch, depth=n_classes, on_value=1.0)

        for i in range(n_text):
            preds = sess.run(self.nodes['pred'], feed_dict={x: x_hot.eval()})
            keys = np.argmax(preds, axis=1)

            text += "".join(r.decode_array(keys))

            preds = keys.reshape([len(x_batch), len(x_batch[0])])
            x_hot = tf.one_hot(preds, depth=n_classes, on_value=1.0)

        print(text)
        return text

