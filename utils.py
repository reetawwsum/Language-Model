from __future__ import print_function

import os
import zipfile
import tensorflow as tf

class Dataset():
	'''Load dataset'''
	def __init__(self, config, dataset_type):
		self.config = config
		self.dataset_type = dataset_type
		self.file_name = os.path.join(config.dataset_dir, config.dataset)
		self.validation_size = config.validation_size

		self.load_dataset()

	def load_dataset(self):
		self.load()
		train_words, validation_words = self.split()

		if self.dataset_type == 'train_dataset':
			self.data = train_words
		else:
			self.data = validation_words

	def load(self):
		'''Reading dataset as a list of words'''
		with zipfile.ZipFile(self.file_name) as f:
			words = tf.compat.as_str(f.read(f.namelist()[0])).split()

		self.words = words

	def split(self):
		validation_words = self.words[:self.validation_size]
		train_words = self.words[self.validation_size:]

		return train_words, validation_words
