from model import model
from reader import Reader
import tensorflow as tf
import itertools, sys
import numpy as np
import time
# from colorama import Fore, Back, Style
import sys

from Parser import args

print(args)
# import matplotlib.pyplot as plt

FILE_NAME = args.file
# FILE_NAME = './data/test'

tf.set_random_seed(0)

CHECK_POINT = False
TENSORBOARD = args.tensorboard
VALIDATION = True

# variables
batch_size = args.batch_size
# sequence len
sequence_len = args.sequence_len

r = Reader.Reader(batch_size=batch_size, sequence_len=sequence_len)
r.load(FILE_NAME)
# r.read(FILE_NAME,encode_words=False)
n_classes = r.get_unique_words()

try:
    validationR = Reader.Reader(sequence_len=sequence_len)
    validationR.read(FILE_NAME + '/validation.txt')

    X_val, Y_val = validationR.create_training_set()
except:
    VALIDATION = False

x = tf.placeholder(tf.float32, [None, None, n_classes])
y = tf.placeholder(tf.float32,[None, None, n_classes])

model = model.RNN()

pred, cost, train_step, accuracy, correct_pred = model.build(x, y, args.layers, n_classes, args.eta, args.dropout)

sess = tf.InteractiveSession()

saver = tf.train.Saver()

losses = []
accuracies = []
total_loss = 0
total_acc = 0
total_val_loss = 0

n_batches = len(r.data)//(batch_size * sequence_len)

epochs = args.epochs

start_time = time.clock()

print('-------+-------')
print('================')
print(args.layers)
print("Sequence len: {:^8}".format(sequence_len))
print("Learning rate: {:^8}".format(args.eta))
print("Batch size: {:^8}".format(batch_size))
print('Dropout: {}'.format(args.dropout))
print('================')
print("{}".format(FILE_NAME))
print("Data size: {:^8}".format(len(r.data)))
print("N classes: {:^8}".format(n_classes))
print("N batches: {:^8}".format(n_batches))
print('-------+-------')
print("N epochs: {:^8}".format(epochs))
print('Start at: {:^8}'.format(time.strftime("%H:%M:%S")))
print('\n')

X,Y = r.create_training_set()

if(TENSORBOARD):
    writer = tf.summary.FileWriter("output/plot_1", sess.graph)
    writer_val = tf.summary.FileWriter("output/plot_2", sess.graph)

    tf.summary.scalar("loss", cost)

    write_op = tf.summary.merge_all()

if(VALIDATION):
    X_val_hot = tf.one_hot(X_val, depth=n_classes, on_value=1.0)
    Y_val_hot = tf.one_hot(Y_val, depth=n_classes, on_value=1.0)


N_TEXT = 100

sess.run(tf.global_variables_initializer())

try:
    for n in range(epochs):
            for i in range(len(X)//batch_size):
                x_batch,y_batch = r.next(i)

                x_hot = tf.one_hot(x_batch, depth=n_classes, on_value=1.0)
                y_hot = tf.one_hot(y_batch, depth=n_classes, on_value=1.0)

                _, loss, acc = sess.run([ train_step, cost, accuracy ], feed_dict={x: x_hot.eval(), y: y_hot.eval()})

                if(VALIDATION):
                    _, val_loss = sess.run([pred, cost], feed_dict={x: X_val_hot.eval(),y: Y_val_hot.eval()})

                if(args.verbose):
                    print('Done batch {}'.format(i))
                    print('Iter: {}'.format(i * (n +1)))
                    print('Loss: {}'.format(loss))

                    if(VALIDATION):
                        print('Val Loss: {}'.format(val_loss))
                        total_val_loss += val_loss

                    if(TENSORBOARD):
                        summary = sess.run(write_op, {cost: loss})
                        writer.add_summary(summary, i * (n +1) )
                        writer.flush()

                        summary = sess.run(write_op, {cost: val_loss})
                        writer_val.add_summary(summary, i * (n +1) )
                        writer_val.flush()
                    #
                total_loss += loss
                total_acc += acc

            print('--------')
            print('Epoch: {}'.format(n))

            print('AVG Loss:  {0:.4f}'.format(total_loss/(len(X)//batch_size)))
            if(VALIDATION):
                print('AVG Val Loss:  {0:.4f}'.format(total_val_loss/(len(X)//batch_size)))
            print('AVG Acc: {0:.4f}'.format(total_acc/(len(X)//batch_size)))

            # model.generate(x,y, 'T', sess, n_classes, r)
            total_loss = 0
            total_acc = 0
            total_val_loss = 0

            if(CHECK_POINT):
                save_path = saver.save(sess, "/tmp/model-{}.ckpt".format(time.strftime("%H:%M:%S")))
                print("Model saved in file: %s" % save_path)

except KeyboardInterrupt:
    pass

if(TENSORBOARD):
    writer.close()
x_batch,y_batch = r.next(0)

model.generate(x, y, 'T', sess, n_classes, r)

save_path = saver.save(sess, "/tmp/model.ckpt")
print("Model saved in file: %s" % save_path)

finish_time = time.clock()
total_time = finish_time - start_time

print('----------')
print('Finish After: {0:.4f}s'.format(total_time))
print('Finish at: {:^8}'.format(time.strftime("%H:%M:%S")))
print('Iterations per second: {0:.4f}'.format(total_time/epochs))

exit(1)
