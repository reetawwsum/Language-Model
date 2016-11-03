from __future__ import print_function

import os
import zipfile
import tensorflow as tf
from collections import Counter

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
		self.build_vocabulary()
		self.convert_words_to_wordids()
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

	def build_vocabulary(self):
		counter = Counter(self.words)
		count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
		words, _ = list(zip(*count_pairs))
		self.words2id = dict(zip(words, range(len(words))))

	def convert_words_to_wordids(self):
		self.wordids = [self.words2id[word] for word in self.words]

	def split(self):
		validation_words = self.wordids[:self.validation_size]
		train_words = self.wordids[self.validation_size:]

		return train_words, validation_words
