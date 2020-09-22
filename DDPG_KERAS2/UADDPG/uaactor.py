import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Input, Dense, Reshape, LSTM, Lambda, BatchNormalization, GaussianNoise, Flatten, Dropout
from keras import Sequential
import keras

import keras.backend as K
import re


def insert_layer_nonseq(model, layer_regex, insert_layer_factory, insert_layer_name=None, position='after'):
    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                # network_dict['input_layers_of'][layer_name].append(layer.name)
                network_dict['input_layers_of'].update({layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
        {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                       for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            new_layer = insert_layer_factory()
            if insert_layer_name:
                new_layer.name = insert_layer_name
            else:
                new_layer.name = '{}_{}'.format(layer.name,
                                                new_layer.name)
            x = new_layer(x)
            print('New layer: {} Old layer: {} Type: {}'.format(new_layer.name,
                                                                layer.name, position))
            if position == 'before':
                x = layer(x)
        else:
            print(layer.name)
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model

        if layer_name in model.output_names:
            model_outputs.append(x)

    print("model.inputs", model.inputs)
    print("model_outputs", model_outputs)
    return Model(inputs=model.inputs, outputs=model_outputs)

def dropout_layer_factory():
    return Dropout(rate=0.2, name='dropout')

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

        self.model = self.network_epistemic()
        self.target_model = self.network_epistemic()
        self.adam_optimizer = self.optimizer()
        self.aware_aleatoric = aware_aleatoric
        self.aware_epistemic = aware_epistemic

    def network_epistemic(self):
        model = Sequential()
        model.add(Dense(256, activation='relu', name='layer1', input_shape=(self.env_dim)))
        model.add(Dropout(self.dropout_p))
        model.add(GaussianNoise(1.0))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', name='layer2'))
        model.add(Dropout(self.dropout_p))
        model.add(GaussianNoise(1.0))

        model.add(Dense(self.act_dim, activation='tanh', name='layer3', kernel_initializer=RandomUniform()))
        model.add(Lambda(lambda i: i * self.act_range))

        print(model)
        model.training=True

        return model

    def predict(self, state):

        for i in range(3):

            action = self.model.predict(np.expand_dims(state, axis=0))
            print("plr", i, action)

        # action = self.model.predict(np.expand_dims(state, axis=0))
        # print("plt", action)

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
