from model import model
from reader import Reader
import tensorflow as tf
import numpy as np
import time
import sys
import os
from Parser import args
from logger import Logger
# import matplotlib.pyplot as plt

FILE_NAME = args.file
# FILE_NAME = './data/test'

tf.set_random_seed(0)

CHECK_POINT = args.checkpoint
TENSORBOARD = args.tensorboard
VALIDATION = True
START = time.strftime("%H:%M:%S")
OUTPUT_FILE = "output-{}.txt".format(START)
# variables
batch_size = args.batch_size
# sequence len
sequence_len = args.sequence_len

r = Reader.Reader(batch_size=batch_size, sequence_len=sequence_len)
r.load(FILE_NAME)
# r.read(FILE_NAME,encode_words=False)
n_classes = r.get_unique_words()

logger = Logger.Logger(True, True)


x = tf.placeholder(tf.int64, [None, None], name='X')
y = tf.placeholder(tf.int64,[None, None], name='Y')

x = tf.one_hot(x, depth=n_classes)
y = tf.one_hot(y, depth=n_classes)

model = model.RNN.from_args(x, y, args)

pred, cost, train_step, accuracy = model.build()

initial_state, state = model.get_states()

config = tf.ConfigProto()

config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config=config)

saver = tf.train.Saver()

losses = []
accuracies = []
total_loss = 0
total_acc = 0
total_val_loss = 0

n_batches = len(r.data)//(batch_size * sequence_len)

epochs = args.epochs

logger.log(logger.get_model_definition(args,r,n_batches))

if(TENSORBOARD):
    writer = tf.summary.FileWriter("output/plot_1", sess.graph)
    writer_val = tf.summary.FileWriter("output/plot_2", sess.graph)

    tf.summary.scalar("loss", cost)

    write_op = tf.summary.merge_all()

if(CHECK_POINT):
    DIR_NAME = "./checkpoint-{}".format(START)
    os.makedirs(DIR_NAME)

N_TEXT = 100

sess.run(tf.global_variables_initializer())

start_time = time.clock()

n = 0

last_state = None

try:
    for x_batch, y_batch, epoch in r.create_iter(epochs):

        feed_dict = {'X:0': x_batch, 'Y:0': y_batch, 'pkeep:0': 0.8}

        if(last_state != None):
            feed_dict[initial_state] = last_state

        _, last_state, loss, acc = sess.run([ train_step, state, cost, accuracy  ], feed_dict=feed_dict)

        # logger.log("Doing batch {} of the {} epoch\n".format(n % n_batches, epoch))

        n = n + 1

        total_loss += loss
        total_acc += acc

        if(n % n_batches == 0 and n > 0):
            avg_loss = total_loss/n_batches
            avg_acc = total_acc/n_batches
        #
        #
            preds = sess.run(pred, feed_dict=feed_dict)
            keys = np.argmax(preds, axis=1)
            pred_text = ""
            # pred_text = model.generate(x, y, 'T', sess, n_classes, r, 1000)
            text  = ""

            feed_dict = {'X:0': r.val_data['X'],'Y:0': r.val_data['Y'], 'pkeep:0': 1.0}

            _, val_loss = sess.run([pred, cost], feed_dict=feed_dict)
        #
            logger.log(logger.get_current_train_info(epoch, avg_loss, avg_acc, val_loss, pred_text, text))
        #

            if (n / 2 % n_batches == 0):
                text = model.generate(x, y, 'T', sess, n_classes, r, 1000)
                logger.log("{}\n".format(text))
        #
            total_loss = 0
            total_acc = 0
            total_val_loss = 0
        #
            if(CHECK_POINT):
                if(n/5 % n_batches == 0):
                    save_path = saver.save(sess, "{}/model-{}.ckpt".format(DIR_NAME,time.strftime("%H:%M:%S")))
                    print("Model saved in file: %s" % save_path)


except KeyboardInterrupt:
    pass

if(TENSORBOARD):
    writer.close()

text = model.generate(x, y, 'T', sess, n_classes, r, 3000)
logger.log(text)
save_path = saver.save(sess, "/tmp/model.ckpt")
print("Model saved in file: %s" % save_path)

finish_time = time.clock()
total_time = finish_time - start_time

print('----------')
print('Finish after: {0:.4f}s'.format(total_time))
print('Finish at: {:^8}'.format(time.strftime("%H:%M:%S")))
print('Epoch per second: {0:.4f}'.format(total_time/epochs))

exit(1)
