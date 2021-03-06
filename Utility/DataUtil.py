import codecs
import fnmatch
import os
import random
import sys
import re
from collections import *
import nltk
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import logging

import csv
# for long CSV
maxInt = sys.maxsize
decrement = True
while decrement:
	# decrease the maxInt value by factor 10
	# as long as the OverflowError occurs.
	decrement = False
	try:
		csv.field_size_limit(maxInt)
	except OverflowError:
		maxInt = int(maxInt / 10)
		decrement = True



class MyError(Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)


class TextDataHandler:
	_force_read_input = False
	_word_to_id = {}
	_id_to_word = {}
	_words = []
	_unknown_terms = set()
	_unknown_terms_freq = 0

	def __init__(self, all_doc_path, save_dir):

		files = Utilities.recursive_glob(all_doc_path, "*")
		Utilities.create_file_dir(save_dir)
		# base_name = os.path.basename(os.path.normpath(all_doc_path))
		# word_to_id
		word_to_id_pickle = os.path.join(save_dir, 'word_to_id.pkl')
		self._word_to_id = self._maybe_pickling(word_to_id_pickle, self._build_dict_from_a_set_of_file, files)

		# id_to_word
		id_to_word_pickle = os.path.join(save_dir, 'id_to_word.pkl')
		self._id_to_word = self._maybe_pickling(id_to_word_pickle, self._build_reverse_dict, self._word_to_id)

	def _maybe_pickling(self, pickle_name, func, *func_param):
		if os.path.exists(pickle_name) and not self._force_read_input:
			# You may override by setting force=True.
			print('%s already present - Skipping pickling.' % pickle_name)
			try:
				with open(pickle_name, 'rb') as f:
					data = pickle.load(f)
			except Exception as e:
				print('Unable to process data from', pickle_name, ':', e)
				raise
		else:
			data = func(*func_param)
			self.pickling(pickle_name, data)
		return data

	def pickling(self, pickle_file_name, data):
		print('Pickling %s' % pickle_file_name)
		try:
			with open(pickle_file_name, 'wb') as f:
				pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
		except Exception as e:
			logging.error('Unable to save data to', pickle_file_name, ':', e)

	@staticmethod
	def clean_str(string):
		"""
		string cleanin
		"""
		string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
		string = re.sub(r"\'s", " \'s", string)
		string = re.sub(r"\'ve", " \'ve", string)
		string = re.sub(r"n\'t", " n\'t", string)
		string = re.sub(r"\'re", " \'re", string)
		string = re.sub(r"\'d", " \'d", string)
		string = re.sub(r"\'ll", " \'ll", string)
		string = re.sub(r",", " , ", string)
		string = re.sub(r"!", " ! ", string)
		string = re.sub(r"\(", " \( ", string)
		string = re.sub(r"\)", " \) ", string)
		string = re.sub(r"\?", " \? ", string)
		string = re.sub(r"\s{2,}", " ", string)
		#       string = re.sub(r"[^A-Za-z0-9]", " ", string)
		return string.strip().lower()

	def read_words(self, filename):
		"""
		Tokenization using NLTK
		"""
		with tf.gfile.GFile(filename, "r") as f:
			return nltk.word_tokenize(TextDataHandler.clean_str(f.read()))

	def _build_dict_from_a_set_of_file(self, filenames):
		counter = Counter()
		file_cnt = len(filenames)
		cnt = 0
		for filename in filenames:
			if cnt % 500 == 0:
				print(str(cnt) + " out of " + str(file_cnt) + " has been processed")
			words = self.read_words(filename)
			counter.update(words)
			cnt += 1
		count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
		# if vocab_size == -1:
		# 	vocab_size = len(count_pairs)
		# words, _ = list(zip(*count_pairs[:vocab_size]))
		words, _ = list(zip(*count_pairs))
		word_to_id = dict(zip(words, range(1, len(words) + 1)))
		return word_to_id

	def get_vocab_size(self):
		if not self._word_to_id.values:
			raise MyError('first build the dictionary...')
		return len(self._word_to_id) + 1  # unknown

	def truncate_vocab(self, new_vocab_size):
		if new_vocab_size < 0:
			print("New Vocab Size:", self.get_vocab_size())
			return
		self._word_to_id = OrderedDict(sorted(self._word_to_id.items(), key=lambda item: item[1])[:new_vocab_size - 1])
		self._id_to_word = self._build_reverse_dict(self._word_to_id)
		print("New Vocab Size:", self.get_vocab_size())

	def _build_reverse_dict(self, original_dict):
		if not original_dict:
			raise MyError('first build the dictionary...')
		reverse_dict = dict(zip(original_dict.values(), original_dict.keys()))
		return reverse_dict

	def get_id_of_word(self, word):
		#     return self._word_to_id.get(word, 0) # dictionary['UNK'] = 0
		if word in self._word_to_id:
			return self._word_to_id[word]
		else:  # dictionary['UNK'] = 0
			self._unknown_terms.add(word)
			self._unknown_terms_freq += 1
			return 0

	def get_word_of_id(self, i):
		if i == 0:
			word = 'UNK'
		elif i in self._id_to_word:
			word = self._id_to_word[i]
		else:
			print(i)
			raise MyError('unknown_id')
		return word

	def word_list_to_id_list(self, words):
		return [self.get_id_of_word(word) for word in words]

	def id_list_to_word_list(self, ids):
		return [self.get_word_of_id(i) for i in ids]

	def get_vocab_dict(self):
		return self._word_to_id.copy()

	def get_rev_vocab_dict(self):
		return self._id_to_word.copy()

	def get_vocab(self):
		if not self._words:
			self._words = list(self._word_to_id)
		return self._words

	def get_one_hot_vector(self, wordIds_vec):
		vocab_size = self.get_vocab_size()
		vec = (np.arange(vocab_size) == np.array(wordIds_vec)[:, None]).astype(np.float32)
		return np.array(vec)

	def get_ids_from_one_hot_vector(self, one_hot_Vec):
		#to be implimented
		return None

	def get_binary_vector(self, wordIds_vec):
		vec = np.zeros(self.get_vocab_size(), dtype=int)
		inices = list(set(wordIds_vec))
		vec[inices] = 1
		return np.array(vec)

	def get_ids_from_binary_vector(self, binary_vector):
		return np.nonzero(binary_vector)

	def get_freq_vector(self, wordIds_vec):
		vec = np.zeros(self.get_vocab_size(), dtype=int)
		counter = Counter(wordIds_vec)
		vec[list(counter.keys())] = list(counter.values())
		return np.array(vec)

	def get_ids_from_freq_vector(self, freq_vector):
		# return [[freq_vector[i]] * i for i in np.nonzero(freq_vector)]
		return None




class BatchCreator4RNN:
	def __init__(self, tensor, batch_size, seq_length):
		self.tensor = tensor
		self.batch_size = batch_size
		self.seq_length = seq_length
		self.create_batches()
		self.pointer = 0

	# abcd -> x:abc y:bcd
	def create_batches(self):
		self.num_batches = int(self.tensor.size / (self.batch_size *
																							 self.seq_length))
		if self.num_batches == 0:
			assert False, "Not enough data. Make seq_length and batch_size small."

		self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
		xdata = self.tensor
		ydata = np.copy(self.tensor)

		ydata[:-1] = xdata[1:]
		ydata[-1] = xdata[0]
		self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
		self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

	def next_batch(self):
		x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
		self.pointer += 1
		return x, y

	def reset_batch_pointer(self):
		self.pointer = 0


class Retrieval_Data_Util(object):
	doc_query_pairs = []

	def __init__(self, runres, qrel):
		with codecs.open(qrel, "r", encoding='utf-8', errors='ignore') as f:
			csv_reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
			for row in csv_reader:
				if row[3] == '1':
					self.doc_query_pairs.append((row[2], row[0]))
			f.close()
		self._runRes = runres

	def get_rel_qd(self):
		d = [tup[0] for tup in self.doc_query_pairs]
		q = [tup[1] for tup in self.doc_query_pairs]
		return d, q

	def get_pseudo_rel_qd(self, top_k):
		d = []
		q = []
		with codecs.open(self._runRes, "r", encoding='utf-8', errors='ignore') as f:
			csv_reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
			for row in csv_reader:
				if int(row[3]) <= top_k:
					d.append(row[2])
					q.append(row[0])
			f.close()
		return d, q

	def get_label(self, d, q):
		return (d, q) in self.doc_query_pairs



class Utilities():
	@staticmethod
	def shufflize(data, label):
		assert len(data) == len(label)
		perm = np.random.permutation(len(data))
		return data[perm], data[perm]

	@staticmethod
	def recursive_glob(treeroot, pattern):
		results = []
		for base, dirs, files in os.walk(treeroot):
			goodfiles = fnmatch.filter(files, pattern)
			results.extend(os.path.join(base, f) for f in goodfiles)
		return results

	@staticmethod
	def create_file_dir(filename):
		if not os.path.exists(os.path.dirname(filename)):
			try:
				os.makedirs(os.path.dirname(filename))
			except OSError as exc:  # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise

	@staticmethod
	def sample_distribution(distribution):
		"""Sample one element from a distribution assumed to be an array of normalized
    probabilities.
    """
		r = random.uniform(0, 1)
		s = 0
		for i in range(len(distribution)):
			s += distribution[i]
			if s >= r:
				return i
		return len(distribution) - 1

	@staticmethod
	def random_distribution(vector_size):
		"""Generate a random column of probabilities."""
		b = np.random.uniform(0.0, 1.0, size=[1, vector_size])
		return b / np.sum(b, 1)[:, None]

	@staticmethod
	def accuracy(predictions, labels):
		return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
						/ predictions.shape[0])
