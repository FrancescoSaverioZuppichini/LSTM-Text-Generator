import tensorflow as tf
from parser import args
from model import model



def generate_from(checkpoint_name, start_text, n_text):

    tf.reset_default_graph()

    n_classes = 255

    my_model = model.RNN.from_args(args, n_classes)

    pred, preds, cost, train_step, accuracy = my_model.build()

    initial_state, state = my_model.get_states()


    with tf.Session() as sess:

            saver = tf.train.Saver()

            saver.restore(sess, tf.train.latest_checkpoint("./checkpoints-{}".format(checkpoint_name)))

            text = my_model.generate(start_text, sess, interactive=True, n_text=n_text)

            return text
    # print(text)

# inputs = ['T','SCENE I ', 'ACT I ', 'CLEOPATRA', 'DUKE ', 'SCENE I	Lugano.', 'SCENE II The world','P',
#           'PAULO', 'DEEP', 'SHAKESPEARE', '	A TALE\n\n', 'DR.','SCENE II', 'SCENE III ', 'KING ', 'DRAGON',
#           'NOBODY', 'CASSIUS', 'MIRKO ', 'DARIO']
#
# for start_text in inputs:
#     generate_from('shakespeare',start_text)
#     interactive shell

# generate_from('shakespeare','\n\tPROUD TO BE\n\n\n\tDRAMATIS PERSONAE\n\n')
# while(True):
#     start_text = input("\nType Something: ")
#
#     generate_from('shakespeare',start_text)