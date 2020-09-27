import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from contrib import adf
print(adf.__file__)


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

def keep_variance(x, min_variance):
    return x + min_variance




class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Critic, self).__init__()
        self.nb_states = nb_states
        self.fc1 = nn.Linear(nb_states, hidden1)

        self.fc2 = nn.Linear(hidden1 + nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(torch.cat([out, a], 1))
        out = self.relu(out)
        out = self.fc3(out)
        return out


class UAActor(nn.Module):
    def __init__(self, nb_states, nb_actions, is_target = True, hidden1=400, hidden2=300, init_w=3e-3):
        super(UAActor, self).__init__()

        self.min_variance = 1e-4

        self.keep_variance_fn = lambda x: keep_variance(x, min_variance=self.min_variance)
        self._noise_variance = 1e-4
        self.dropout_p = 0.1
        self.dropout_n = 1
        self.is_target = is_target

        self.linear1 = adf.Linear(nb_states, hidden1, keep_variance_fn=self.keep_variance_fn)
        self.linear2 = adf.Linear(hidden1, hidden2, keep_variance_fn=self.keep_variance_fn)
        self.linear3 = adf.Linear(hidden2, nb_actions, keep_variance_fn=self.keep_variance_fn)

        self.ReLU = adf.ReLU(keep_variance_fn=self.keep_variance_fn)

        self.dropout = adf.Dropout(p=self.dropout_p, keep_variance_fn=self.keep_variance_fn)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.linear1.weight.data = fanin_init(self.linear1.weight.data.size())
        self.linear2.weight.data = fanin_init(self.linear2.weight.data.size())
        self.linear3.weight.data.uniform_(-init_w, init_w)


    def forward(self, x):
        inputs_mean = x
        inputs_var = torch.zeros_like(inputs_mean) + self._noise_variance

        inputs_mean_nor = (inputs_mean - torch.min(inputs_mean)) / (torch.max(inputs_mean)- torch.min(inputs_mean))

        tmp_input = inputs_mean_nor+0.1, inputs_var
        for _ in range(self.dropout_n):

            out = self.linear1(*tmp_input)
            out = self.ReLU(*out)
            out = self.dropout(*out)
            out = self.linear2(*out)
            out = self.ReLU(*out)
            out = self.dropout(*out)
            out = self.linear3(*out)

        return out


class UACritic(nn.Module):
    def __init__(self, nb_states, nb_actions, dropout_n=3, dropout_p=0.2, hidden1=400, hidden2=300, init_w=3e-3):
        super(UACritic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc1_dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden1 + nb_actions, hidden2)
        self.fc2_dropout = nn.Dropout(p=dropout_p)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
        self.dropout_n = dropout_n

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(torch.cat([out, a[:-1]], 1))
        out = self.relu(out)
        out = self.fc3(out)
        return out

    def forward_with_dropout(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc1_dropout(out)
        out = self.fc2(torch.cat([out, a[:-1]], 1))
        out = self.relu(out)
        out = self.fc2_dropout(out)
        out = self.fc3(out)
        return out

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out