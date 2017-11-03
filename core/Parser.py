import argparse

parser = argparse.ArgumentParser(description='LSTM implementation for text generation')

parser.add_argument('--file', help='A text file', default='./data/dataset.txt', type=str)
parser.add_argument('--dir', help='A directory of text files', type=int)
parser.add_argument('--batch_size', help='The batch size', default=32, type=int)
parser.add_argument('--sequence_len', help='The sequence len', default=200, type=int)
parser.add_argument('--eta', help='The learning rate', default=0.001, type=float)
parser.add_argument('--layers', help='List of layer info', default=[216], type=lambda s: [int(item) for item in s.split(',')])
parser.add_argument('--dropout', help='The number of layers', default=False, type=bool)
parser.add_argument('--epochs', help='Number of epochs', default=1000, type=int)
parser.add_argument('--tensorboard', help='Setup tensorboard', default=False, type=bool)
parser.add_argument('--verbose', help='Output at each iteration', default=False, type=bool)

args = parser.parse_args()

