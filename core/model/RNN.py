import tensorflow as tf
from tensorflow.contrib import rnn
from reader import Reader
import numpy as np

r = Reader.Reader()
r.read('../story.txt',encode_words=True)

symbols_number = r.get_unique_words()
r.init_batch(3,1,random=True)

dictionary = r.data_set
vocab_size = symbols_number

# Parameters
learning_rate = 0.001
training_iters = 50000
display_step = 1000
n_input = 3

sess = tf.InteractiveSession()
# number of units in RNN cell
n_hidden = 512

# tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])

w = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
b = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}


def RNN(x, w, b):
    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,n_input,1)

    # 1-layer LSTM with n_hidden units.
    rnn_cell = rnn.BasicLSTMCell(n_hidden)
    # rnn_cell =rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], w['out']) + b['out']

pred = RNN(x, w, b)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sess.run(tf.global_variables_initializer())

for n in range(50000):
    loss_total = 0
    acc_total = 0
    inputs,targets = r.next()
    # print(inputs,targets)
    if(inputs == None):
        r.init_batch(3, 1,random=True)
        inputs,targets = r.next()

    inputs_keys = [r.encode(input) for input in inputs]

    inputs_keys =  np.reshape(np.array(inputs_keys), [-1, n_input, 1])

    one_hot_encoding = np.zeros([symbols_number],dtype=float)
    one_hot_encoding[r.encode(targets[0])] = 1.0

    one_hot_encoding = np.reshape(one_hot_encoding, [1, -1])

    _, acc, loss, onehot_pred = sess.run([optimizer, accuracy, cost, pred],
                                            feed_dict={x: inputs_keys, y: one_hot_encoding})

    loss_total += loss
    acc_total += acc

    if(n % 100 == 0):
        print("---------------")
        print("Iterations: {}".format(n))
        print("Loss: {}".format(loss))
        print("AVG Loss: {}".format(loss_total/100))
        print("AVG acc: {}".format(acc_total/100))
        out =  r.decode(int(tf.argmax(onehot_pred, 1).eval()))
        # print(onehot_pred)
        # print(one_hot_encoding)
        # sess.run(accuracy, feed_dict={x:onehot_pred, y :one_hot_encoding})

        print("{}, {} - {}".format(inputs, out, targets))
        loss_total = 0
        acc_total = 0


inputs,targets = r.next(random=True)

# inputs = ['ulisse','vidi','ponte']

inputs_keys = [r.encode(input) for input in inputs]

keys = np.reshape(np.array(inputs_keys), [-1, n_input, 1])

one_hot_encoding = np.zeros([symbols_number], dtype=float)
one_hot_encoding[r.encode(targets[0])] = 1.0

one_hot_encoding = np.reshape(one_hot_encoding, [1, -1])
text = inputs

for _ in range(100):
    print(inputs_keys)
    onehot_pred = sess.run(pred, feed_dict={x: keys})
    onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
    pred_word = r.decode(onehot_pred_index)

    print(pred_word)
    inputs.append(pred_word)

    inputs_keys[0] = inputs_keys[1]
    inputs_keys[1] = inputs_keys[2]
    inputs_keys[2] = r.encode(pred_word)
    keys =  np.reshape(np.array(inputs_keys), [-1, n_input, 1])

print(inputs)