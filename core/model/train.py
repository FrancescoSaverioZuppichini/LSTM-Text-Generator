import time

import tensorflow as tf
from model import model

# from parser import args
from logger import Logger
from reader import Reader


# import matplotlib.pyplot as plt
def train_model(args):
    tf.set_random_seed(0)
    # global flags
    CHECK_POINT = args.checkpoint
    TENSORBOARD = args.tensorboard
    START = time.strftime("%H:%M:%S")
    OUTPUT_FILE = "output-{}.txt".format(START)
    # variables from args
    batch_size = args.batch_size
    sequence_len = args.sequence_len
    epochs = args.epochs
    # Init reader
    r = Reader.Reader(batch_size=batch_size, sequence_len=sequence_len)
    r.load(args.file)
    # derived variables
    n_classes = 255
    n_batches = len(r.data)//(batch_size * sequence_len)
    # create a logger to print/save informations
    logger = Logger.Logger(True, True)

    # n_classes_var = tf.Variable(n_classes, dtype=tf.int64, name="n_classes")

    # instance model
    my_model = model.RNN.from_args(args, n_classes)
    # build model graph
    pred,preds, cost, train_step, accuracy = my_model.build()
    # fetch the states variables
    initial_state, state = my_model.get_states()

    sess = tf.InteractiveSession()

    start_time = time.clock()

    total_loss = 0
    total_acc = 0

    n = 0

    last_state = None

    logger.log(logger.get_model_definition(args,r,n_batches))

    saver = tf.train.Saver(max_to_keep=20)

    sess.run(tf.global_variables_initializer())


    for x_batch, y_batch, epoch in r.create_iter(epochs):

        feed_dict = {'X:0': x_batch, 'Y:0': y_batch, 'pkeep:0': args.dropout}

        if(last_state != None):
            # update state
            feed_dict[initial_state] = last_state

        _, last_state, loss,  acc = sess.run([ train_step, state, cost, accuracy  ], feed_dict=feed_dict)

        n = n + 1

        total_loss += loss
        total_acc += acc

        if(n % n_batches == 0 and n > 0):

            avg_loss = total_loss/n_batches
            avg_acc = total_acc/n_batches

            feed_dict = {'X:0': r.val_data['X'], 'Y:0': r.val_data['Y'], 'pkeep:0': 1.0}
            _, val_loss = sess.run([pred, cost], feed_dict=feed_dict)

            logger.log(logger.get_current_train_info(epoch, avg_loss, avg_acc, val_loss))

            if (n / 2 % n_batches == 0):
                text = my_model.generate('T', sess, 1000)
                logger.log("{}\n".format(text))

            total_loss = 0
            total_acc = 0


            if(CHECK_POINT):
                save_path = saver.save(sess, "checkpoints-{}/model.ckpt".format(START), global_step=epoch)
                print("Model saved in file: %s" % save_path)


    finish_time = time.clock()
    total_time = finish_time - start_time

    print('----------')
    print('Finish at: {:^8}'.format(time.strftime("%H:%M:%S")))

