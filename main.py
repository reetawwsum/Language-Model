import tensorflow as tf

from utils import *

flags = tf.app.flags
flags.DEFINE_integer('epochs', 10000, 'Epochs to train')
flags.DEFINE_integer('batch_size', 100, 'Size of training batch')
flags.DEFINE_integer('num_unrollings', 10, 'Input sequence length')
flags.DEFINE_integer('validation_size', 1000, 'Size of validation dataset')
flags.DEFINE_string('dataset', 'text8.zip', 'Name of dataset file')
flags.DEFINE_string('batch_dataset_type', 'train_dataset', 'Dataset used for generating training batches')
flags.DEFINE_string('dataset_dir', 'data', 'Directory name for the dataset')
FLAGS = flags.FLAGS

def main(_):
	train_batches = BatchGenerator(FLAGS)
	train_data = train_batches.next()
	print(train_data.shape)

if __name__ == '__main__':
	tf.app.run()
