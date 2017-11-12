# import nltk
from random import randint
import numpy as np
import tensorflow as tf
import os

class Reader():

    def __init__(self, sequence_len=64,batch_size=32):
        self.data_set = {}
        self.data_set_inv = {}
        self.data = []
        self.sequence_len = sequence_len
        self.batch_size = batch_size
        self.training = {'X' : [], 'Y' :[]}


    def load(self, path):
        """
        Load a provided path, it will check if is a directory
        or a normal txt file and select the correct behavior
        :param path: Path to a file/dir
        :return:
        """
        isFile = os.path.isfile(path)

        if(isFile):
            self.read_file(path)
        else:
            self.read_from_dir(path)

    def read_from_dir(self, path):
        """
        Read and store each file from a given directory
        :param path: Path to the directory with the files
        :return:
        """
        files = [file for file in os.listdir(path) if file.endswith(".txt")]

        words = ""
        for file in files:
            words += self.read(path + '/' + file)

        self.create_data_set(words)

    def read(self, path):
        """
        Read and return the words from a given file
        :param path: Path to the file we want to read
        :return:
        """
        with open(path,'r',encoding='utf8') as file:
            words = file.read()

        print('Loading {}'.format(os.path.basename(path)))

        return words

    def read_file(self, path):
        """
        Read and store the words from a given file
        :param path: Path to the file we want to read
        :return:
        """
        words = self.read(path)
        self.create_data_set(words)

    def create_data_set(self, words):

        self.data = words

        self.data_encoded = [self.encode(char) for char in words]
        # train 80% and val 10%
        offset = len(self.data_encoded) // 100 * 90

        self.train_data = self.data_encoded[:offset]

        self.val_data = {'X':[], 'Y':[]}

        self.val_data['X'], self.val_data['Y'] = self.create_data(self.data_encoded[offset:], 1, self.sequence_len)

    def decode_array(self, array):
        return [self.decode(index) for index in array]

    def encode_array(self, array):
        return [self.encode(index) for index in array]

    def decode(self,index):
        return chr(index)

    def encode(self,word):
        return ord(word)

    def get_unique_words(self):
        return len(self.data_set.keys())

    def print_info(self):
        print('Data size: {}'.format(len(self.data)))
        print('Uniques symbols: {}'.format(self.get_unique_words()))

    def create_data(self, raw_data, batch_size, sequence_size):
        data = np.array(raw_data)
        data_len = data.shape[0]
        nb_batches = (data_len - 1) // (batch_size * sequence_size)
        assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
        rounded_data_len = nb_batches * batch_size * sequence_size
        xdata = np.reshape(data[0:rounded_data_len], [batch_size, nb_batches * sequence_size])
        ydata = np.reshape(data[1:rounded_data_len + 1], [batch_size, nb_batches * sequence_size])

        return xdata, ydata

    # copied from https://github.com/martin-gorner/tensorflow-rnn-shakespeare/blob/master/my_txtutils.py#L99
    def create_iter(self, nb_epochs):
        """
        Divides the data into batches of sequences so that all the sequences in one batch
        continue in the next batch. This is a generator that will keep returning batches
        until the input data has been seen nb_epochs times. Sequences are continued even
        between epochs, apart from one, the one corresponding to the end of raw_data.
        The remainder at the end of raw_data that does not fit in an full batch is ignored.
        :param raw_data: the training text
        :param batch_size: the size of a training minibatch
        :param sequence_size: the unroll size of the RNN
        :param nb_epochs: number of epochs to train on
        :return:
            x: one batch of training sequences
            y: on batch of target sequences, i.e. training sequences shifted by 1
            epoch: the current epoch number (starting at 0)
        """
        raw_data = self.train_data
        batch_size = self.batch_size
        sequence_size = self.sequence_len
        data = np.array(raw_data)
        data_len = data.shape[0]
        # using (data_len-1) because we must provide for the sequence shifted by 1 too
        nb_batches = (data_len - 1) // (batch_size * sequence_size)
        assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."

        xdata, ydata =  self.create_data(raw_data, batch_size, sequence_size)

        for epoch in range(nb_epochs):
            for batch in range(nb_batches):
                x = xdata[:, batch * sequence_size:(batch + 1) * sequence_size]
                y = ydata[:, batch * sequence_size:(batch + 1) * sequence_size]
                x = np.roll(x, -epoch, axis=0)  # to continue the text from epoch to epoch (do not reset rnn state!)
                y = np.roll(y, -epoch, axis=0)
                yield x, y, epoch

