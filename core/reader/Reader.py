import nltk
from random import randint

class Reader():

    def __init__(self):
        self.data_set = {}
        self.data_set_inv = {}
        self.data = []
        self.pointer = {'input': 0, 'target':0, 'prev':0 , 'step': 0}

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
        if(encode_words):
            words = nltk.word_tokenize(words)

        self.data = words
        # little trick, create a set that will get rid of duplicated
        unique_chars = list(set(words))

        self.data_set = {k: v for k,v in enumerate(unique_chars)}
        self.data_set_inv = {k: v for v,k in enumerate(unique_chars)}

    def decode(self,index):
        return self.data_set[index]

    def encode(self,word):
        return self.data_set_inv[word]

    def init_batch(self,input,target, random=False, step=1):

        if(input > len(self.data)):
            raise Exception('Input size cannot be larger than the data set.')

        self.pointer['input'] = input
        self.pointer['target'] = target
        self.pointer['prev'] = 0
        self.pointer['step'] = step


        if(random):
            self.pointer['prev'] = randint(0, len(self.data) - input - target - 1)

    def next(self, random=False):
        p = self.pointer['prev']
        i = self.pointer['input']
        t = self.pointer['target']
        s = self.pointer['step']

        if(random):
            p = randint(0, len(self.data) - i - t)

        lower = p + i
        upper = lower + t

        if lower >= len(self.data):
            return (None, None)


        inputs, targets = ( self.data[p: lower], self.data[lower: upper] )

        self.pointer['prev'] = self.pointer['prev'] + s

        return inputs, targets

    def get_unique_words(self):
        return len(self.data_set.keys())
# little test
#
# r = Reader()
# r.read('../gesu_2.txt',encode_words=True)
# # print(r.decode(1))
# # print(r.encode('the'))
# # print(r.get_unique_words())
# r.init_batch(3,1)
# # print(r.next(random=True))
# # #
# # # text = ""
# # #
# for _ in range(20):
#     (i,t) = r.next()
#     if(i == None):
#         break
# #     print('---------------')
#     print(i)
#     print(t)
