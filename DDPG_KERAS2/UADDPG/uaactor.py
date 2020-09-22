import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model as Model
from keras.layers import Input, Dense, Reshape, LSTM, Lambda, BatchNormalization, GaussianNoise, Flatten, Dropout
from keras import Sequential
import keras

import warnings
warnings.filterwarnings("ignore")

import math
# from scipy.misc import logsumexp
import numpy as np

from keras.regularizers import l2
from keras import Input
from keras.layers import Dropout
from keras.layers import Dense
# from keras import Model as Model2

import keras.backend as K
import re


class UAActor:
    """ Actor Network for the DDPG Algorithm
    """

    def __init__(self, inp_dim, out_dim, act_range, lr, tau, aware_aleatoric, aware_epistemic, dropout_n, dropout_p):
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.act_range = act_range
        self.tau = tau
        self.lr = lr
        self.dropout_n = dropout_n
        self.dropout_p = dropout_p
        self.batch_size = 128
        self.n_hidden = [256, 128, 64]
        self.model = self.network_epistemic()
        self.target_model = self.network_epistemic()
        self.adam_optimizer = self.optimizer()
        self.aware_aleatoric = aware_aleatoric
        self.aware_epistemic = aware_epistemic

    def network_epistemic(self):

        inputs = Input(shape=(self.env_dim))
        inter = Dropout(self.dropout_p)(inputs, training=True)
        inter = Dense(self.n_hidden[0], activation='relu')(inter)
        for i in range(len(self.n_hidden) - 1):
            inter = Dropout(self.dropout_p)(inter, training=True)
            inter = Dense(self.n_hidden[i + 1], activation='relu')(inter)
        inter = Dropout(self.dropout_p)(inter, training=True)
        outputs = Dense(self.act_dim)(inter)
        model = Model(inputs, outputs)

        # model.compile(loss='mean_squared_error', optimizer='adam')

        return model

    def predict(self, state):
        print("input state:", state)
        for i in range(3):

            action = self.model.predict(np.expand_dims(state, axis=0))
            print("plr", i, action)

        # action = self.model.predict(np.expand_dims(state, axis=0))

        return action

    def target_predict(self, inp):
        """ Action prediction (target network)
        """
        return self.target_model.predict(inp)

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        self.target_model.set_weights(target_W)

    def train(self, states, actions, grads):
        """ Actor Training
        """
        self.adam_optimizer([states, grads])

    def optimizer(self):
        """ Actor Optimizer
        """
        action_gdts = K.placeholder(shape=(None, self.act_dim))
        params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -action_gdts)
        grads = zip(params_grad, self.model.trainable_weights)
        # return K.function([self.model.input, action_gdts], [tf.compat.v1.train.AdamOptimizer(self.lr).apply_gradients(grads)])
        return K.function([self.model.input, action_gdts], [tf.train.AdamOptimizer(self.lr).apply_gradients(grads)][1:])

    def save(self, path):
        self.model.save_weights(path + '_actor.h5')

    def load_weights(self, path):
        self.model.load_weights(path)


"""
BACKUP FOR Uncertainty-Aware Actor
"""

# class UAActor:
#     """ Actor Network for the DDPG Algorithm
#     """
#
#     def __init__(self, inp_dim, out_dim, act_range, lr, tau, aware_aleatoric, aware_epistemic, dropout_n, dropout_p):
#         self.env_dim = inp_dim
#         self.act_dim = out_dim
#         self.act_range = act_range
#         self.tau = tau
#         self.lr = lr
#         self.dropout_n = dropout_n
#         self.dropout_p = dropout_p
#
#         self.model = self.network_epistemic()
#         self.target_model = self.network_epistemic()
#         self.adam_optimizer = self.optimizer()
#         self.aware_aleatoric = aware_aleatoric
#         self.aware_epistemic = aware_epistemic
#
#     def network_epistemic(self):
#         model = Sequential()
#         model.add(Dense(256, activation='relu', name='layer1', input_shape=(self.env_dim)))
#         model.add(Dropout(self.dropout_p))
#         model.add(GaussianNoise(1.0))
#         model.add(Flatten())
#         model.add(Dense(128, activation='relu', name='layer2'))
#         model.add(Dropout(self.dropout_p))
#         model.add(GaussianNoise(1.0))
#
#         model.add(Dense(self.act_dim, activation='tanh', name='layer3', kernel_initializer=RandomUniform()))
#         model.add(Lambda(lambda i: i * self.act_range))
#
#         print(model)
#         model.training=True
#
#         return model
#
#     def predict(self, state):
#
#         for i in range(3):
#
#             action = self.model.predict(np.expand_dims(state, axis=0))
#             print("plr", i, action)
#
#         # action = self.model.predict(np.expand_dims(state, axis=0))
#         # print("plt", action)
#
#         return action
#
#     def target_predict(self, inp):
#         """ Action prediction (target network)
#         """
#         return self.target_model.predict(inp)
#
#     def transfer_weights(self):
#         """ Transfer model weights to target model with a factor of Tau
#         """
#         W, target_W = self.model.get_weights(), self.target_model.get_weights()
#         for i in range(len(W)):
#             target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
#         self.target_model.set_weights(target_W)
#
#     def train(self, states, actions, grads):
#         """ Actor Training
#         """
#         self.adam_optimizer([states, grads])
#
#     def optimizer(self):
#         """ Actor Optimizer
#         """
#         action_gdts = K.placeholder(shape=(None, self.act_dim))
#         params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -action_gdts)
#         grads = zip(params_grad, self.model.trainable_weights)
#         # return K.function([self.model.input, action_gdts], [tf.compat.v1.train.AdamOptimizer(self.lr).apply_gradients(grads)])
#         return K.function([self.model.input, action_gdts], [tf.train.AdamOptimizer(self.lr).apply_gradients(grads)][1:])
#
#     def save(self, path):
#         self.model.save_weights(path + '_actor.h5')
#
#     def load_weights(self, path):
#         self.model.load_weights(path)
