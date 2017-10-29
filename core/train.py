from model import model
from reader import Reader
import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np

FILE_NAME = './story.txt'

r = Reader.Reader()
r.read(FILE_NAME,encode_words=True)
# r.read(FILE_NAME)

n_output = r.get_unique_words()
n_hidden = 516

n_input = 4
n_target = 1

MAX_ITER = 50000
N_TEXT = 30

chunk_size = 1

r.init_batch(n_input, n_target)

sess = tf.InteractiveSession()

x = tf.placeholder("float", [None, n_input, chunk_size])
y = tf.placeholder("float", [None, n_output])

pred, cost, train_step, accuracy = model.RNN(x, y, n_input, n_hidden, n_output)

print('------------')
print('Using: {}'.format(FILE_NAME))
print('Unique words: {}'.format(n_output))
print('\n')

saver = tf.train.Saver()

saver.restore(sess, "/tmp/model.ckpt")
#
# sess.run(tf.global_variables_initializer())


losses = []

total_loss = 0
total_acc = 0

n_batches = len(r.data)//n_input

for n in range(0):

    inputs,targets = r.next()

    if (inputs == None):
        r.init_batch(n_input, n_target,random=False)
        inputs, targets = r.next()

    X = np.array([r.encode(input) for input in inputs])

    X = np.reshape(X, [-1,n_input,1])

    Y = np.zeros([n_output], dtype=float)
    Y[r.encode(targets[0])] = 1.0

    Y = np.reshape(Y, [1, -1])

    output, loss, acc = sess.run([ train_step, cost, accuracy] ,feed_dict={x: X, y: Y})

    losses.append(loss)

    total_loss += loss
    total_acc += acc

    # if(n % 100):
    #     out = sess.run(pred, feed_dict={x: X, y: Y})
    #     out = r.decode(int(tf.argmax(out, 1).eval()))
    #     print("{}, {} - {}".format(inputs, out, targets))

    if(n % 1000 == 0 and n > 0):

        out = sess.run(pred,feed_dict={x: X, y: Y})
        out =  r.decode(int(tf.argmax(out, 1).eval()))

        print('--------')
        print('Iterations: {}'.format(n))
        print('AVG Loss: {}'.format(total_loss/(1000 )))
        print('AVG Acc: {}'.format(total_acc/(1000 )))
        # print("{}, {} - {}".format(inputs, out, targets))

        # if(total_loss < 0.001):
        #     break
        total_loss = 0
        total_acc = 0

    # if(n % 100 == 0):
    #     # if (total_loss < 0.001):
    #     #     break
    #     print('Read dataset {} times'.format(n))
            # if(n % 1000 == 0):
        #     print('Iterations: {}'.format(n))

            # print("{}, {} - {}".format(inputs, out, targets))

save_path = saver.save(sess, "/tmp/model.ckpt")

print("Model saved in file: %s" % save_path)

inputs,targets = r.next(random=True)

keys = [r.encode(input) for input in inputs]

X = np.reshape(np.array(keys), [-1, n_input, 1])

Y = np.zeros([n_output], dtype=float)
Y[r.encode(targets[0])] = 1.0

Y = np.reshape(Y, [1, -1])

text = inputs

for _ in range(N_TEXT):
    # print(keys)
    onehot_pred = sess.run(pred, feed_dict={x: X})
    onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
    pred_word = r.decode(onehot_pred_index)

    inputs.append(pred_word)
    # text+= pred_word
    keys[0] = keys[1]
    keys[1] = keys[2]
    keys[2] = r.encode(pred_word)

    X =  np.reshape(np.array(keys), [-1, n_input, 1])

print(" ".join(text))
# print(text)
# losses = np.mean(np.array(losses).reshape(-1, 10), 1)
# plt.plot(losses)
# plt.show()