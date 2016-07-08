import numpy as np
import tensorflow as tf

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from Utility.DataToBinaryVec import DataLoader


class LogReg(object):

  def __init__(self):
    self.d_loader = DataLoader()
    self.vector_size = self.d_loader.d_handler.get_vocab_size()
    self.train_dataset, self.train_labels, self.valid_dataset, \
    self.valid_labels, self.test_dataset, self.test_labels = self.d_loader.get_ttv()

  def logistic_regression_using_simple_gradient_descent(self):
    print("creating the computational graph...")
    graph = tf.Graph()
    with graph.as_default():
      tf_train_dataset = tf.constant(self.train_dataset)
      tf_train_labels = tf.constant(self.train_labels)
      tf_valid_dataset = tf.constant(self.valid_dataset)
      tf_test_dataset = tf.constant(self.test_dataset)

      weights = tf.Variable(
        tf.truncated_normal([self.vector_size, self.vector_size]))
      biases = tf.Variable(tf.zeros([self.vector_size]))

      def LinRegModel(dataset, weightsw, biases):
        return tf.matmul(dataset, weights) + biases

      logits = LinRegModel(tf_train_dataset, weights, biases)
      # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
      loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits,
                                                                    tf_train_labels))  # Measures the probability error in discrete classification tasks in which each class is independent and not mutually exclusive.
      optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
      train_prediction = tf.nn.softmax(logits)
      valid_prediction = tf.nn.softmax(LinRegModel(tf_valid_dataset, weights, biases))
      test_prediction = tf.nn.softmax(LinRegModel(tf_test_dataset, weights, biases))
      with tf.name_scope('accuracy'):
        pre = tf.placeholder("float", shape=[None, self.vector_size])
        lbl = tf.placeholder("float", shape=[None, self.vector_size])
        accuracy = tf.reduce_mean(tf.cast(tf.nn.sigmoid_cross_entropy_with_logits(pre, lbl), "float"))

    num_steps = 801
    with tf.Session(graph=graph) as session:
      tf.initialize_all_variables().run()
      print('Initialized')
      for step in range(num_steps):
        _, l, predictions = session.run([optimizer, loss, train_prediction])
        if (step % 100 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print('Training accuracy: %.1f%%' % session.run(accuracy,
                                                          feed_dict={pre: predictions, lbl: self.train_labels}))
          print('Validation accuracy:  %.3f%%' % session.run(accuracy,
                                                             feed_dict={pre: valid_prediction.eval(), lbl: self.valid_labels}))
          print('Test accuracy:  %.3f%%' % session.run(accuracy,
                                                       feed_dict={pre: test_prediction.eval(), lbl: self.test_labels}))


  def logistic_regression_using_stochastic_gradient_descent(self):
    batch_size = 100
    print("creating the computational graph...")
    graph = tf.Graph()
    with graph.as_default():
      tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, self.vector_size))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, self.vector_size))
      tf_valid_dataset = tf.constant(self.valid_dataset)
      tf_test_dataset = tf.constant(self.test_dataset)

      weights = tf.Variable(
        tf.truncated_normal([self.vector_size, self.vector_size]))
      biases = tf.Variable(tf.zeros([self.vector_size]))

      def LinRegModel(dataset, weightsw, biases):
        return tf.matmul(dataset, weights) + biases

      logits = LinRegModel(tf_train_dataset, weights, biases)
      # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
      loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits,
                                                                    tf_train_labels))  # Measures the probability error in discrete classification tasks in which each class is independent and not mutually exclusive.
      optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
      train_prediction = tf.nn.softmax(logits)
      valid_prediction = tf.nn.softmax(LinRegModel(tf_valid_dataset, weights, biases))
      test_prediction = tf.nn.softmax(LinRegModel(tf_test_dataset, weights, biases))

      with tf.name_scope('accuracy'):
        pre = tf.placeholder("float", shape=[None, self.vector_size])
        lbl = tf.placeholder("float", shape=[None, self.vector_size])
        accuracy = tf.reduce_mean(tf.cast(tf.nn.sigmoid_cross_entropy_with_logits(pre, lbl), "float"))

    num_steps = 800
    print('running the session...')
    with tf.Session(graph=graph) as session:
      tf.initialize_all_variables().run()
      print('Initialized')
      for step in range(num_steps):
        offset = (step * batch_size) % (self.train_labels.shape[0] - batch_size)
        batch_data = self.train_dataset[offset:(offset + batch_size), :]
        batch_labels = self.train_labels[offset:(offset + batch_size), :]

        print('-' * 80)
        for vec in batch_labels:
          print('.' * 200)
          print(self.get_words(vec))

        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.3f%%" % session.run(accuracy,
                                                           feed_dict={pre: predictions, lbl: batch_labels}))
          # self.print_words(predictions, batch_labels)
          print('Validation accuracy:  %.3f%%' % session.run(accuracy,
                                                             feed_dict={pre: valid_prediction.eval(), lbl: self.valid_labels}))
      print('Test accuracy:  %.3f%%' % session.run(accuracy,
                                                   feed_dict={pre: test_prediction.eval(), lbl: self.test_labels}))
      self.print_words(test_prediction.eval(),self.test_labels)


  def print_words(self, preds, labels):
    for pred, label in zip(preds,labels):
      label_ids = self.d_loader.d_handler.get_ids_from_binary_vector(label)[0]
      pred_ids = np.argsort(np.negative(pred))[:label_ids.size]
      # pred_ids = np.argsort(pred)[(-(label_ids.size)):][::-1]
      # print(label_ids)
      # print(pred_ids)
      print(self.d_loader.d_handler.id_list_to_word_list(pred_ids))
      print(self.d_loader.d_handler.id_list_to_word_list(label_ids))
      # break

  def get_words(self,vect):
    ids = self.d_loader.d_handler.get_ids_from_binary_vector(vect)[0]
    return self.d_loader.d_handler.id_list_to_word_list(ids)

LG = LogReg()
LG.logistic_regression_using_stochastic_gradient_descent()
# LG.logistic_regression_using_simple_gradient_descent()

print("done...")
