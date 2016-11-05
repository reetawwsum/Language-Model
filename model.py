import tensorflow as tf

from ops import *
from utils import *

class Model():
	'''RNN LSTM neural network'''
	def __init__(self, config):
		self.config = config
		self.batch_size = config.batch_size
		self.num_unrollings = config.num_unrollings
		self.vocabulary_size = config.vocabulary_size
		self.num_units = config.num_units
		self.num_hidden_layers = config.num_hidden_layers
		self.learning_rate = config.learning_rate

		self.build_model()

	def inference(self):
		# Creating LSTM layer
		cell = tf.nn.rnn_cell.LSTMCell(self.num_units)
		cell = tf.nn.rnn_cell.DropoutWrapper(cell, self.input_keep_prob, self.output_keep_prob)
		cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_hidden_layers)

		# Creating Unrolled LSTM
		hidden, _ = tf.nn.dynamic_rnn(cell, self.data, dtype=tf.float32)

		# Creating output at each time step
		reshaped_hidden = tf.reshape(hidden, (-1, self.num_units))
		logit = tf.matmul(reshaped_hidden, self.weight) + self.bias

		self.logit = logit

	def loss_op(self):
		loss = tf.nn.seq2seq.sequence_loss_by_example([self.logit], [self.target], [tf.ones((self.batch_size * self.num_unrollings))])

		self.loss = tf.reduce_sum(loss) / self.batch_size

	def train_op(self):
		optimizer = tf.train.AdamOptimizer(self.learning_rate)

		self.optimizer = optimizer.minimize(self.loss)

	def build_model(self):
		self.graph = tf.Graph()

		with self.graph.as_default():
			# Creating placeholder for data
			data, self.target = placeholder_input(self.batch_size, self.num_unrollings)

			# Converting data to embeddings
			self.data = embeddings(data, self.vocabulary_size, self.num_units)

			# Creating placeholder for LSTM dropout
			self.input_keep_prob = self.output_keep_prob = placeholder_dropout()

			# Creating variables for output layer
			self.weight = weight_variable([self.num_units, self.vocabulary_size])
			self.bias = bias_variable([self.vocabulary_size])

			# Builds the graph that computes inference
			self.inference()	

			# Adding loss op to the graph
			self.loss_op()

			# Adding train op to the grpah
			self.train_op()
