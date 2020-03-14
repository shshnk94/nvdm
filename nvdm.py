"NVDM Tensorflow implementation by Yishu Miao"""
from __future__ import print_function

import numpy as np
import tensorflow as tf
import math
import os
import utils as utils

np.random.seed(0)
tf.set_random_seed(0)

flags = tf.app.flags
flags.DEFINE_string('data_dir', 'data/20news', 'Data dir path.')
flags.DEFINE_float('learning_rate', 5e-5, 'Learning rate.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('n_hidden', 500, 'Size of each hidden layer.')
flags.DEFINE_integer('n_topic', 50, 'Size of stochastic vector.')
flags.DEFINE_integer('n_sample', 1, 'Number of samples.')
flags.DEFINE_integer('vocab_size', 2000, 'Vocabulary size.')
flags.DEFINE_boolean('test', False, 'Process test data.')
flags.DEFINE_string('non_linearity', 'tanh', 'Non-linearity of the MLP.')
FLAGS = flags.FLAGS

class NVDM(object):
    """ Neural Variational Document Model -- BOW VAE.
    """
    def __init__(self, 
                 vocab_size,
                 n_hidden,
                 n_topic, 
                 n_sample,
                 learning_rate, 
                 batch_size,
                 non_linearity):
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_topic = n_topic
        self.n_sample = n_sample
        self.non_linearity = non_linearity
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.x = tf.placeholder(tf.float32, [None, vocab_size], name='input')
        self.mask = tf.placeholder(tf.float32, [None], name='mask')  # mask paddings

        # encoder
        with tf.variable_scope('encoder'): 
          self.enc_vec = utils.mlp(self.x, [self.n_hidden], self.non_linearity)
          self.mean = utils.linear(self.enc_vec, self.n_topic, scope='mean')
          self.logsigm = utils.linear(self.enc_vec, 
                                     self.n_topic, 
                                     bias_start_zero=True,
                                     matrix_start_zero=True,
                                     scope='logsigm')
          self.kld = -0.5 * tf.reduce_sum(1 - tf.square(self.mean) + 2 * self.logsigm - tf.exp(2 * self.logsigm), 1)
          self.kld = self.mask*self.kld  # mask paddings
        
        with tf.variable_scope('decoder'):
          if self.n_sample ==1:  # single sample
            eps = tf.random_normal((batch_size, self.n_topic), 0, 1)
            doc_vec = tf.multiply(tf.exp(self.logsigm), eps) + self.mean
            logits = tf.nn.log_softmax(utils.linear(doc_vec, self.vocab_size, scope='projection'))
            self.recons_loss = -tf.reduce_sum(tf.multiply(logits, self.x), 1)
          # multiple samples
          else:
            eps = tf.random_normal((self.n_sample*batch_size, self.n_topic), 0, 1)
            eps_list = tf.split(0, self.n_sample, eps)
            recons_loss_list = []
            for i in range(self.n_sample):
              if i > 0: tf.get_variable_scope().reuse_variables()
              curr_eps = eps_list[i]
              doc_vec = tf.multiply(tf.exp(self.logsigm), curr_eps) + self.mean
              logits = tf.nn.log_softmax(utils.linear(doc_vec, self.vocab_size, scope='projection'))
              recons_loss_list.append(-tf.reduce_sum(tf.multiply(logits, self.x), 1))
            self.recons_loss = tf.add_n(recons_loss_list) / self.n_sample

        self.objective = self.recons_loss + self.kld

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        fullvars = tf.trainable_variables()

        enc_vars = utils.variable_parser(fullvars, 'encoder')
        dec_vars = utils.variable_parser(fullvars, 'decoder')

        enc_grads = tf.gradients(self.objective, enc_vars)
        dec_grads = tf.gradients(self.objective, dec_vars)

        self.optim_enc = optimizer.apply_gradients(zip(enc_grads, enc_vars))
        self.optim_dec = optimizer.apply_gradients(zip(dec_grads, dec_vars))

def get_topic_diversity(probabilities, topk):

    num_topics = probabilities.shape[0]
    list_w = np.zeros((num_topics, topk))

    for k in range(num_topics):
        idx = probabilities[k,:].argsort()[-topk:]
        list_w[k,:] = idx
    n_unique = len(np.unique(list_w))
    diversity = n_unique / (topk * num_topics)

    return diversity

def get_document_frequency(data, wi, wj=None):

  if wj is None:
    D_wi = 0
    for l in range(len(data)):
      doc = data[l].keys()
      if len(doc) == 1: 
        continue
      else:
        if wi in doc:
          D_wi += 1
    
    return D_wi

  D_wj = 0
  D_wi_wj = 0

  for l in range(len(data)):
    doc = data[l].keys()
    if wj in doc:
      D_wj += 1
    if wi in doc:
      D_wi_wj += 1

  return D_wj, D_wi_wj 

def get_topic_coherence(probabilities, tokens):

  #top_10_words = probabilities.argsort(axis=1)[::-1][:, :10]
  D = len(tokens)

  coherence = []
  num_topics = probabilities.shape[0]

  for k in range(num_topics):

    top_words = list(probabilities[k].argsort()[-11:][::-1])
    
    coherence_k = 0
    counter = 0

    for i, word in enumerate(top_words):
      D_wi = get_document_frequency(tokens, word)
      j = i + 1
      tmp = 0

      while j < len(top_words) and j > i:
        D_wj, D_wi_wj = get_document_frequency(tokens, word, top_words[j])
        if not D_wi or not D_wj or not D_wi_wj:
          print(word, top_words[j], D_wi, D_wj, D_wi_wj)
        if not D_wi_wj:
          f_wi_wj = -1
        else:
          f_wi_wj = -1 + (np.log(D_wi) + np.log(D_wj)  - 2.0 * np.log(D)) / (np.log(D_wi_wj) - np.log(D))

        tmp += f_wi_wj
        j += 1
        counter += 1
            
      coherence_k += tmp 

    coherence.append(coherence_k)
    
  return np.mean(coherence) / counter

def evaluate(model, training_data, dataset, dataset_count, data_batches, session, step, epoch=None, summaries=None, writer=None):

  loss_sum = 0.0
  kld_sum = 0.0
  ppx_sum = 0.0
  word_count = 0
  doc_count = 0

  for idx_batch in data_batches:

    data_batch, count_batch, mask = utils.fetch_data(dataset, 
                                                     dataset_count, 
                                                     idx_batch, 
                                                     FLAGS.vocab_size)
    input_feed = {model.x.name: data_batch, model.mask.name: mask}
    loss, kld = session.run([model.objective, model.kld], input_feed)
    
    loss_sum += np.sum(loss)
    kld_sum += np.sum(kld) / np.sum(mask)  
    word_count += np.sum(count_batch)
    count_batch = np.add(count_batch, 1e-12)
    ppx_sum += np.sum(np.divide(loss, count_batch))
    doc_count += np.sum(mask) 

  print_ppx = np.exp(loss_sum / word_count)
  print_ppx_perdoc = np.exp(ppx_sum / doc_count)
  print_kld = kld_sum/len(data_batches)
  
  probabilities = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder/projection/Matrix:0')[0]
  coherence = get_topic_coherence(probabilities.eval(session), training_data)
  diversity = get_topic_diversity(probabilities.eval(session), 25)
    
  if step == 'val':
 
    weight_summaries = session.run(summaries)
    writer.add_summary(weight_summaries, epoch)

    saver = tf.train.Saver()
    save_path = saver.save(session, "./results/model.ckpt")
    print("Model saved in path: %s" % save_path)
    print('| Epoch dev: {:d} |'.format(epoch+1)) 

  print('| Perplexity: {:.9f}'.format(print_ppx),
        '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
        '| KLD: {:.5}'.format(print_kld))

  with open('./results/report.csv', 'a') as handle:
    handle.write(str(print_ppx_perdoc) + ',' + str(coherence) + ',' + str(diversity) + '\n')

def get_summaries(sess):

  weights = tf.trainable_variables()
  values = sess.run(weights)

  weight_summaries = []
  for weight, value in zip(weights, values):
    weight_summaries.append(tf.summary.histogram(weight.name, value))

  return tf.summary.merge(weight_summaries) 

def train(sess, model, 
          train_url, 
          test_url, 
          batch_size, 
          training_epochs=1000, 
          alternate_epochs=10):

  """train nvdm model."""
  train_set, train_count = utils.data_set(train_url)
  test_set, test_count = utils.data_set(test_url)
  # hold-out development dataset
  dev_set = test_set[:50]
  dev_count = test_count[:50]

  dev_batches = utils.create_batches(len(dev_set), batch_size, shuffle=False)
  test_batches = utils.create_batches(len(test_set), batch_size, shuffle=False)
  
  summaries = get_summaries(sess) 
  writer = tf.summary.FileWriter('./results/logs/', sess.graph)
  
  for epoch in range(training_epochs):

    train_batches = utils.create_batches(len(train_set), batch_size, shuffle=True)

    for switch in range(0, 2):

      if switch == 0:
        optim = model.optim_dec
        print_mode = 'updating decoder'
      else:
        optim = model.optim_enc
        print_mode = 'updating encoder'

      for i in range(alternate_epochs):
        loss_sum = 0.0
        ppx_sum = 0.0
        kld_sum = 0.0
        word_count = 0
        doc_count = 0
        for idx_batch in train_batches:
          data_batch, count_batch, mask = utils.fetch_data(
          train_set, train_count, idx_batch, FLAGS.vocab_size)
          input_feed = {model.x.name: data_batch, model.mask.name: mask}
          _, (loss, kld) = sess.run((optim, 
                                    [model.objective, model.kld]),
                                    input_feed)
          loss_sum += np.sum(loss)
          kld_sum += np.sum(kld) / np.sum(mask) 
          word_count += np.sum(count_batch)
          # to avoid nan error
          count_batch = np.add(count_batch, 1e-12)
          # per document loss
          ppx_sum += np.sum(np.divide(loss, count_batch)) 
          doc_count += np.sum(mask)
        print_ppx = np.exp(loss_sum / word_count)
        print_ppx_perdoc = np.exp(ppx_sum / doc_count)
        print_kld = kld_sum/len(train_batches)
        print('| Epoch train: {:d} |'.format(epoch+1), 
               print_mode, '{:d}'.format(i),
               '| Corpus ppx: {:.5f}'.format(print_ppx),  # perplexity for all docs
               '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),  # perplexity for per doc
               '| KLD: {:.5}'.format(print_kld))
    
    #Validation
    evaluate(model, train_set, dev_set, dev_count, dev_batches, sess, 'val', epoch, summaries, writer)

def main(argv=None):
    if FLAGS.non_linearity == 'tanh':
      non_linearity = tf.nn.tanh
    elif FLAGS.non_linearity == 'sigmoid':
      non_linearity = tf.nn.sigmoid
    else:
      non_linearity = tf.nn.relu

    nvdm = NVDM(vocab_size=FLAGS.vocab_size,
                n_hidden=FLAGS.n_hidden,
                n_topic=FLAGS.n_topic, 
                n_sample=FLAGS.n_sample,
                learning_rate=FLAGS.learning_rate, 
                batch_size=FLAGS.batch_size,
                non_linearity=non_linearity)

    sess = tf.Session()
    init = tf.initialize_all_variables()

    sess.run(init)

    train_url = os.path.join(FLAGS.data_dir, 'train.feat')
    test_url = os.path.join(FLAGS.data_dir, 'test.feat')
    
    if not FLAGS.test:
      train(sess, nvdm, train_url, test_url, FLAGS.batch_size)
    
    else:
      #Test
      test_set, test_count = utils.data_set(test_url)
      test_batches = utils.create_batches(len(test_set), FLAGS.batch_size, shuffle=False)
      saver.restore(sess, "./results/model.ckpt")
      print("Model restored.")
      
      #Training data
      train_set, train_count = utils.data_set(train_url)
      evaluate(model, train_set, test_set, test_count, test_batches, sess, 'test') 

if __name__ == '__main__':
    tf.app.run()
