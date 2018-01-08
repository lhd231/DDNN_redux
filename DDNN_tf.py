import tensorflow as tf
import numpy as np
import cPickle, gzip
import math




class local_site():


    def __init__(self,ID,n_input,hidden_layers,seed=45):
        self.current_gradients = []

        self.value_dict = {}
        self.value_dict['ID'] = ID
        self.value_dict['n_input']  = n_input
        self.value_dict['hidden_layers']  = hidden_layers
        self.value_dict['seed']  = seed

        self.value_dict['X'] = tf.placeholder(tf.float32, [None, self.value_dict['n_input']],name = self.value_dict['ID']+'_w')
        self.value_dict['Y'] = tf.placeholder(tf.float32, [None, self.value_dict['hidden_layers'][-1]], name = self.value_dict['ID']+'_b')

        self.value_dict['weights'] = []
        self.value_dict['biases'] = []

        self.value_dict['pred'] = 0

    def feed(self,d,l):
        self.value_dict['data'] = d
        self.value_dict['labels'] = l

    def get_data(self,range = None):
        if range is None:
            return self.value_dict['data'],self.value_dict['labels']
        else:
            return self.value_dict['data'][range], self.value_dict['labels'][range]
    def get(self):



        current_size = self.value_dict['n_input']

        for l in self.value_dict['hidden_layers']:
            self.value_dict['weights'].append(tf.Variable(tf.random_normal([current_size, l],seed=self.value_dict['seed']),name = self.value_dict['ID']+'_w1'))

            current_size = l

        for b in self.value_dict['hidden_layers']:
            self.value_dict['biases'].append(tf.Variable(tf.random_normal([b],seed=self.value_dict['seed']), name=self.value_dict['ID'] + '_b1'))

        self.value_dict['pred'] = self.value_dict['X']

        for x,b in zip(self.value_dict['weights'],self.value_dict['biases']):
            self.value_dict['pred'] = tf.nn.softmax(tf.matmul(self.value_dict['pred'], x) + b)
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.value_dict['Y'] * tf.log(self.value_dict['pred']), reduction_indices=1))

        self.grads = []

        self.weights_biases = []
        for w,b in zip(self.value_dict['weights'][::-1],self.value_dict['biases'][::-1]):
            self.weights_biases.append(w)
            self.weights_biases.append(b)

        self.grads=tf.gradients(xs=self.weights_biases, ys=self.cost)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.value_dict['pred'],1),tf.argmax(self.value_dict['Y'],1)),tf.float32))
        return self.value_dict['X'],self.value_dict['Y'], self.cost,self.weights_biases, self.grads,self.accuracy

