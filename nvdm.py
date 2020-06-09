"NVDM Tensorflow implementation by Yishu Miao"""
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pickle as pkl
import math
import os
import sys
import time
import utils as utils
from scipy.special import softmax
from sklearn.metrics import pairwise_distances

from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from metrics import get_topic_coherence, get_topic_diversity, get_perplexity


import gc
import psutil
process = psutil.Process(os.getpid())

np.random.seed(0)
tf.set_random_seed(0)

flags = tf.app.flags
flags.DEFINE_string('data_dir', 'data/20news', 'Data dir path.')
flags.DEFINE_string('save_path', './results', 'Output path.')
flags.DEFINE_float('learning_rate', 5e-5, 'Learning rate.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('n_hidden', 500, 'Size of each hidden layer.')
flags.DEFINE_integer('n_topic', 50, 'Size of stochastic vector.')
flags.DEFINE_integer('n_sample', 1, 'Number of samples.')
flags.DEFINE_integer('n_words', 10, 'Number of words to be displayed per topic.')
flags.DEFINE_integer('vocab_size', 2000, 'Vocabulary size.')
flags.DEFINE_boolean('test', False, 'Process test data.')
flags.DEFINE_string('non_linearity', 'tanh', 'Non-linearity of the MLP.')
flags.DEFINE_integer('epochs', 25, 'Number of training epochs.')
flags.DEFINE_string('fold', '', 'Cross validation fold number.')
flags.DEFINE_string('load_from', '', 'Points to model checkpoint')
FLAGS = flags.FLAGS

if FLAGS.test:
    ckpt = FLAGS.load_from
else:
    if FLAGS.fold != '':
        FLAGS.data_dir = os.path.join(FLAGS.data_dir, 'fold{}'.format(FLAGS.fold))
        ckpt = os.path.join(FLAGS.save_path, 'k{}_e{}_lr{}'.format(FLAGS.n_topic, FLAGS.epochs, FLAGS.learning_rate), 'fold{}'.format(FLAGS.fold))
    else:
        ckpt = FLAGS.save_path

process = psutil.Process(os.getpid())

class NVDM(object):

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
            self.doc_vec = tf.multiply(tf.exp(self.logsigm), eps) + self.mean
            logits = tf.nn.log_softmax(utils.linear(self.doc_vec, self.vocab_size, scope='projection'))
            self.recons_loss = -tf.reduce_sum(tf.multiply(logits, self.x), 1)

          # multiple samples
          else:

            eps = tf.random_normal((self.n_sample*batch_size, self.n_topic), 0, 1)
            eps_list = tf.split(0, self.n_sample, eps)

            recons_loss_list = []
            doc_vec_list = []

            for i in range(self.n_sample):
              if i > 0: tf.get_variable_scope().reuse_variables()

              curr_eps = eps_list[i]
              doc_vec = tf.multiply(tf.exp(self.logsigm), curr_eps) + self.mean
              logits = tf.nn.log_softmax(utils.linear(doc_vec, self.vocab_size, scope='projection'))
              recons_loss_list.append(-tf.reduce_sum(tf.multiply(logits, self.x), 1))
              doc_vec_list.append(doc_vec)
            
            self.doc_vec = tf.stack(doc_vec_list, axis=0)
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

def evaluate(model, training_data, training_count, session, step, train_loss=None, epoch=None, summaries=None, writer=None, saver=None):

  #Get theta for the H1.
  data_url = os.path.join(FLAGS.data_dir, 'valid_h1.feat' if step != 'test' else 'test_h1.feat')
  dataset, dataset_count = utils.data_set(data_url)
  data_batches = utils.create_batches(len(dataset), FLAGS.batch_size, shuffle=False)
   
  theta = []
  for idx_batch in data_batches:

    data_batch, count_batch, mask = utils.fetch_data(dataset, dataset_count, idx_batch, FLAGS.vocab_size)
    input_feed = {model.x.name: data_batch, model.mask.name: mask}

    logit_theta = session.run(model.doc_vec, input_feed)
    theta.append(softmax(logit_theta, axis=1)) 

  theta = np.concatenate(theta, axis=0)
  beta = softmax(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder/projection/Matrix:0')[0].eval(session), axis=1)

  #H2 to calculate perplexity.
  data_url = os.path.join(FLAGS.data_dir, 'valid_h2.feat' if step != 'test' else 'test_h2.feat')
  dataset, dataset_count = utils.data_set(data_url)
  data_batches = utils.create_batches(len(dataset), FLAGS.batch_size, shuffle=False)

  test_data = [utils.fetch_data(dataset, dataset_count, idx_batch, FLAGS.vocab_size)[0] for idx_batch in data_batches]
  test_data = np.concatenate(test_data, axis=0)

  perplexity = get_perplexity(test_data, theta, beta)
  coherence = get_topic_coherence(beta, training_data, 'nvdm') if  step == 'test' else np.nan
  diversity = get_topic_diversity(beta, 'nvdm') if step == 'test' else np.nan
    
  if step == 'val':

    #tloss = tf.get_default_graph().get_tensor_by_name('tloss:0') 
    #vppl = tf.get_default_graph().get_tensor_by_name('vppl:0') 

    #weight_summaries = session.run(summaries, feed_dict={tloss: train_loss, vppl: perplexity})
    weight_summaries = summaries.eval(session=session)
    writer.add_summary(weight_summaries, epoch)
    save_path = saver.save(session, ckpt + "/model.ckpt")

    print("Model saved in path: %s" % ckpt)
    print('| Epoch dev: {:d} |'.format(epoch+1)) 

  else:
    
    ## get most used topics
    cnt = 0
    thetaWeightedAvg = np.zeros((1, FLAGS.n_topic))
    data_batches = utils.create_batches(len(training_data), FLAGS.batch_size, shuffle=False)

    for idx_batch in data_batches:

        batch, count_batch, mask = utils.fetch_data(training_data, training_count, idx_batch, FLAGS.vocab_size)
        sums = batch.sum(axis=1)
        cnt += sums.sum(axis=0)

        input_feed = {model.x.name: batch, model.mask.name: mask}
        logit_theta = session.run(model.doc_vec, input_feed)
        theta = softmax(logit_theta, axis=1)
        weighed_theta = (theta.T * sums).T
        thetaWeightedAvg += weighed_theta.sum(axis=0)

    thetaWeightedAvg = thetaWeightedAvg.squeeze() / cnt
    print('\nThe 10 most used topics are {}'.format(thetaWeightedAvg.argsort()[::-1][:10]))

    with open(FLAGS.data_dir + '/vocab.new', 'rb') as f:
      vocab = pkl.load(f)

    topic_indices = list(np.random.choice(FLAGS.n_topic, 10)) # 10 random topics
    print('\n')

    with open(ckpt + '/topics.txt', 'w') as f:
      for k in range(FLAGS.n_topic):
        gamma = beta[k]
        top_words = list(gamma.argsort()[-FLAGS.n_words+1:][::-1])
        topic_words = [vocab[a] for a in top_words]
        f.write(str(k) + ' ' + str(topic_words) + '\n')
        print('Topic {}: {}'.format(k, topic_words))

  with open(ckpt + '/' + step + '_scores.csv', 'a') as handle:
    handle.write(str(perplexity) + ',' + str(coherence) + ',' + str(diversity) + '\n')

def get_summaries(sess):

  weights = tf.trainable_variables()
  values = [sess.graph.get_tensor_by_name(w.name) for w in weights]

  summaries = []
  for weight, value in zip(weights, values):
    summaries.append(tf.summary.histogram(weight.name, value))

  #tloss = tf.placeholder(tf.float64, shape=(), name='tloss')
  #summaries.append(tf.summary.scalar('Training_loss', tloss))

  #vppl = tf.placeholder(tf.float64, shape=(), name='vppl')
  #summaries.append(tf.summary.scalar('Validation_ppl', vppl))

  return tf.summary.merge(summaries) 

def train(sess, model, train_url, batch_size, training_epochs=1000, alternate_epochs=10):

  train_set, train_count = utils.data_set(train_url)

  summaries = get_summaries(sess) 
  saver = tf.train.Saver()
  writer = tf.summary.FileWriter(ckpt + '/logs/', sess.graph)

  sess.graph.finalize()
 
  total_mem = 0
  mem = 0
 
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

          data_batch, count_batch, mask = utils.fetch_data(train_set, train_count, idx_batch, FLAGS.vocab_size)
          input_feed = {model.x.name: data_batch, model.mask.name: mask}
          _, (loss, kld) = sess.run((optim, [model.objective, model.kld]), input_feed)

          #loss, kld = tf.cast(loss, tf.float64), tf.cast(kld, tf.float64)
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
        
    evaluate(model, train_set, train_count, sess, 'val', (loss_sum + kld_sum), epoch, summaries, writer, saver)

    current_mem = process.memory_info().rss / (1024 ** 2)
    total_mem += (current_mem - mem)
    print("Memory increase: {}, Cumulative memory: {}, and current {} in MB".format(current_mem - mem, total_mem, current_mem))
    mem = current_mem
    gc.collect()

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
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

    init = tf.initialize_all_variables()
    sess.run(init)

    train_url = os.path.join(FLAGS.data_dir, 'train.feat')
    
    if not FLAGS.test:
      train(sess, nvdm, train_url, FLAGS.batch_size, FLAGS.epochs)
    
    else:
      #Test

      saver = tf.train.Saver()
      saver.restore(sess, ckpt + "/model.ckpt")
      print("Model restored.")
      
      #Training data
      train_set, train_count = utils.data_set(train_url)
      evaluate(nvdm, train_set, train_count, sess, 'test') 

if __name__ == '__main__':
    tf.app.run()
