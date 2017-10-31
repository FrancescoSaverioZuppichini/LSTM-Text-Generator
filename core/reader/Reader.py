# import nltk
from random import randint
import numpy as np

class Reader():

    def __init__(self, sequence_len=64,batch_size=32):
        self.data_set = {}
        self.data_set_inv = {}
        self.data = []

        self.sequence_len = sequence_len
        self.batch_size = batch_size
        self.training = {'X' : [], 'Y' :[]}

    def read(self, path, encode_words=False):
        """
        Read and store the words from a given file
        :param path: Path to the file we want to read
        :return:
        """
        with open(path,'r',encoding='utf8') as file:
            words = file.read()

            self.create_data_set(words, encode_words)

    def create_data_set(self, words, encode_words):
        # if(encode_words):
            # words = nltk.word_tokenize(words)

        self.data = words
        # little trick, create a set that will get rid of duplicated
        unique_chars = list(set(words))

        self.data_set = {k: v for k,v in enumerate(unique_chars)}
        self.data_set_inv = {k: v for v,k in enumerate(unique_chars)}

        self.data_encoded = [self.encode(char) for char in self.data]

    def create_set(self, data, start, end, step):
        len_data = len(self.data)

        S = np.array([data[start:len_data - end]])
        S = S.T

        S = S[:len_data//step * step]

        return S.reshape([-1,step])


    def create_training_set(self):
        step  = self.sequence_len

        X, Y = self.create_set(self.data_encoded, 0, 1, step), self.create_set(self.data_encoded, 1, 0, step)

        self.training = {'X' : X, 'Y' :Y}

        return X, Y

    def decode_array(self, array):

        return [self.decode(index) for index in array]

    def decode(self,index):
        return self.data_set[index]

    def encode(self,word):
        return self.data_set_inv[word]

    def next(self, p):
        X = self.training['X']
        Y = self.training['Y']

        return X[p: p + self.batch_size], Y[p : p + self.batch_size]

    def get_unique_words(self):
        return len(self.data_set.keys())
# little test
# #
# r = Reader(sequence_len=5,batch_size=2)
# r.read('../gesu_2.txt',encode_words=False)
# # # #
# X, Y = r.create_training_set()
# print(''.join(r.decode_array(X[0])))
# print(''.join(r.decode_array(Y[0])))

# print(X.shape)
# print(len(r.data))
# for _ in range(10):
#     for i in range(len(X)//2):
#         print(r.next(i))
# # #
# for _ in range(20):
#     (i,t) = r.next()
# # #
#     print(i,t)
#     # print(len(i), len(t))
