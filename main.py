import tensorflow as tf

from utils import *

flags = tf.app.flags
flags.DEFINE_integer('validation_size', 1000, 'Size of validation dataset')
flags.DEFINE_string('dataset', 'text8.zip', 'Name of dataset file')
flags.DEFINE_string('batch_dataset_type', 'train_dataset', 'Dataset used for generating training batches')
flags.DEFINE_string('dataset_dir', 'data', 'Directory name for the dataset')
FLAGS = flags.FLAGS

def main(_):
	dataset = Dataset(FLAGS, FLAGS.batch_dataset_type)
	data = dataset.data

if __name__ == '__main__':
	tf.app.run()
