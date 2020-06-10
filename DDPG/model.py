
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# from ipdb import set_trace as debug


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class ActorWithDropout(nn.Module):
    def __init__(self, nb_states, nb_actions, sample_n, dropout=0.3, hidden1=400, hidden2=300, init_w=3e-3):
        super(ActorWithDropout, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc1_dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc2_dropout = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.fc3_dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)
        self.sample_n = sample_n

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

    def forward_with_dropout(self, x):
        out = self.fc1(x)
        out = self.fc1_dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.fc2_dropout(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.fc3_dropout(out)
        out = self.tanh(out)
        return out


class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
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
        # debug()
        out = self.fc2(torch.cat([out, a], 1))
        out = self.relu(out)
        out = self.fc3(out)
        return out