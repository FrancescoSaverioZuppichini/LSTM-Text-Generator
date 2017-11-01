from model import model
from reader import Reader
import tensorflow as tf
import itertools, sys
import numpy as np
import time
# from colorama import Fore, Back, Style
import sys


# import matplotlib.pyplot as plt

# FILE_NAME = './dio.txt'
FILE_NAME = './data/shakespeare'

CHECK_POINT = False

# variables
batch_size = 128
# sequence len
n_timesteps = 100
n_target = 100
n_input = n_target
# size of each rnn
n_rnn = [256, 256]
n_layers = 1

learning_rate = 0.001

r = Reader.Reader(batch_size=batch_size, sequence_len=n_timesteps)
r.read_from_dir(FILE_NAME)
# r.read(FILE_NAME,encode_words=False)
n_classes = r.get_unique_words()

x = tf.placeholder(tf.float32, [batch_size, n_timesteps, n_classes])
y = tf.placeholder(tf.float32,[batch_size, n_timesteps, n_classes])


sess = tf.InteractiveSession()

pred, cost, train_step, accuracy, correct_pred = model.RNN(x, y, n_timesteps, n_rnn, n_classes, batch_size)

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

losses = []
accuracies = []

total_loss = 0
total_acc = 0

n_batches = len(r.data)//(batch_size * n_timesteps)

epochs = 20

start_time = time.clock()

print('-------+-------')
print("{}".format(FILE_NAME))
print("Data size: {:^8}".format(len(r.data)))
print("N classes: {:^8}".format(n_classes))
print("batch size: {:^8}".format(batch_size))
print("N batches: {:^8}".format(n_batches))
print('-------+-------')
print("N epochs: {:^8}".format(epochs))
print('Start at: {:^8}'.format(time.strftime("%H:%M:%S")))

X,Y = r.create_training_set()

writer = tf.summary.FileWriter("output", sess.graph)

tf.summary.scalar("loss", cost)
write_op = tf.summary.merge_all()

try:
    for n in range(epochs):
            for i in range(len(X)//batch_size):
                x_batch,y_batch = r.next(i)

                x_hot = tf.one_hot(x_batch, depth=n_classes, on_value=1.0)
                y_hot = tf.one_hot(y_batch, depth=n_classes, on_value=1.0)

                _, loss, acc = sess.run([ train_step, cost, accuracy ], feed_dict={x: x_hot.eval(), y: y_hot.eval()})

                print('Done batch {}'.format(i))
                print('Iter: {}'.format(i * (n +1)))
                print('Cost: {}'.format(loss))
                # summary = sess.run(write_op, {cost: loss})
                # writer.add_summary(summary, i * (n +1) )
                # writer.flush()

                total_loss += loss
                total_acc += acc

            print('--------')
            print('Epoch: {}'.format(n))
            # print('AVG Loss:' + Fore.RED + ' {0:.4f}'.format(total_loss/(len(X)//batch_size)) + Fore.RESET)
            print('AVG Loss:  {0:.4f}'.format(total_loss/(len(X)//batch_size)))
            print('AVG Acc: {0:.4f}'.format(total_acc/(len(X)//batch_size)))

            total_loss = 0
            total_acc = 0

            if(CHECK_POINT):
                save_path = saver.save(sess, "/tmp/model-{}.ckpt".format(time.strftime("%H:%M:%S")))
                print("Model saved in file: %s" % save_path)

except KeyboardInterrupt:
    pass

writer.close()
x_batch,y_batch = r.next(0)

x_hot = tf.one_hot(x_batch, depth=n_classes, on_value=1.0)

for i in range(3):

    preds = sess.run(pred, feed_dict={x: x_hot.eval()})

    keys = np.argmax(preds, axis=1)

    print(''.join(r.decode_array(keys)))
    print('GENERATO FINO A QUI')
    preds = keys.reshape([batch_size, n_timesteps])
    x_hot = tf.one_hot(preds, depth=n_classes, on_value=1.0)

save_path = saver.save(sess, "/tmp/model.ckpt")
print("Model saved in file: %s" % save_path)

N_TEXT = 100

finish_time = time.clock()

total_time = finish_time - start_time

print('----------')
print('Finish After: {0:.4f}s'.format(total_time))
print('Iterations per second: {0:.4f}'.format(total_time/epochs))

exit(1)
