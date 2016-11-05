import tensorflow as tf

from model import *

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 100, 'Size of training batch')
flags.DEFINE_integer('epochs', 10000, 'Epochs to train')
flags.DEFINE_integer('num_units', 200, 'Number of units in LSTM layer')
flags.DEFINE_integer('num_hidden_layers', 2, 'Number of hidden LSTM layers')
flags.DEFINE_integer('num_unrollings', 10, 'Input sequence length')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
flags.DEFINE_integer('vocabulary_size', 10000, 'Vocabulary Size')
flags.DEFINE_integer('validation_size', 1000, 'Size of validation dataset')
flags.DEFINE_boolean('train', True, 'True for training, False for validating')
flags.DEFINE_string('dataset', 'text8.zip', 'Name of dataset file')
flags.DEFINE_string('batch_dataset_type', 'train_dataset', 'Dataset used for generating training batches')
flags.DEFINE_string('dataset_dir', 'data', 'Directory name for the dataset')
FLAGS = flags.FLAGS

def main(_):
	if FLAGS.train:
		model = Model(FLAGS)

if __name__ == '__main__':
	tf.app.run()
