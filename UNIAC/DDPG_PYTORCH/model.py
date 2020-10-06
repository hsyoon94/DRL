import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from contrib import adf
import math

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 10. / np.sqrt(fanin)
    # print("v", v)
    return torch.Tensor(size).uniform_(-v, v)


def keep_variance(x, min_variance):
    # return x + min_variance
    return x

class UAActor(nn.Module):
    def __init__(self, nb_states, nb_actions, is_target = True, hidden1=400, hidden2=300, init_w=3e-3):
        super(UAActor, self).__init__()

        self.min_variance = 1e-4

        self.keep_variance_fn = lambda x: keep_variance(x, min_variance=self.min_variance)
        self._noise_variance = 1e-4
        self.dropout_p = 0.1
        self.dropout_n = 3
        self.is_target = is_target
        self.action_count = 0

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


    def forward(self, x, x_noise):
        inputs_mean = x
        inputs_var = torch.zeros_like(inputs_mean) + self._noise_variance
        # inputs_var = x_noise
        inputs_var = inputs_var + 2

        inputs_mean_nor = (inputs_mean - torch.min(inputs_mean)) / (torch.max(inputs_mean)- torch.min(inputs_mean))

        tmp_input = inputs_mean_nor+0.1, inputs_var
        sampled_output_mean = []
        sampled_output_var = []

        with torch.no_grad():
            for _ in range(self.dropout_n):

                out = self.linear1(*tmp_input)
                out = self.ReLU(*out)
                out = self.dropout(*out)
                out = self.linear2(*out)
                out = self.ReLU(*out)
                out = self.dropout(*out)
                out = self.linear3(*out)

                sampled_output_mean.append(out[0].squeeze())
                sampled_output_var.append(out[1].squeeze())

        for i1 in range(len(sampled_output_mean)):
            sampled_output_mean[i1] = sampled_output_mean[i1].cpu().detach().numpy()
            sampled_output_var[i1] = sampled_output_var[i1].cpu().detach().numpy()

        try:
            np_sampled_output_mean = torch.mean(torch.tensor(sampled_output_mean), axis = 0).cuda()

        except TypeError:
            for i_tmp in range(len(sampled_output_mean)):
                sampled_output_mean[i_tmp] = [sampled_output_mean[i_tmp].tolist()]

            sampled_output_mean = np.array(sampled_output_mean)
            np_sampled_output_mean = torch.mean(torch.tensor(sampled_output_mean), axis=0).cuda()

        mean_var = []

        for i2 in range(len(sampled_output_mean)):
            tmp_mean_var = []
            for j in range(np_sampled_output_mean.shape[0]):
                tmp_mean_var.append( (np_sampled_output_mean[j].cpu().numpy() - sampled_output_mean[i2][j]) ** 2)

            mean_var.append(tmp_mean_var)

        np_sampled_output_var = torch.mean(torch.tensor(sampled_output_var), axis=0) + torch.mean(torch.tensor(mean_var), axis=0)

        if self.is_target is False:
            self.action_count = self.action_count + 1

        if np_sampled_output_var.shape[0]!=64 :
            for i3 in range(len(np_sampled_output_var)):
                # print("var", np_sampled_output_var[i3])
                try:
                    np_sampled_output_var[i3] = math.pow(10, np_sampled_output_var[i3])
                except RuntimeError:
                    print(np_sampled_output_var[i3])

        else:
            for i3 in range(np_sampled_output_var.shape[0]):
                for j3 in range(len(np_sampled_output_var[i3])):
                    np_sampled_output_var[i3][j3] = math.pow(10, np_sampled_output_var[i3][j3])

        return np_sampled_output_mean.cuda(), np_sampled_output_var.cuda()


class UACritic(nn.Module):
    def __init__(self, nb_states, nb_actions, is_target = True, hidden1=400, hidden2=300, init_w=3e-3):
        super(UACritic, self).__init__()

        self.min_variance = 1e-4

        self.keep_variance_fn = lambda x: keep_variance(x, min_variance=self.min_variance)
        self._noise_variance = 1e-4
        self.dropout_p = 0.1
        self.dropout_n = 1
        self.is_target = is_target

        self.linear1 = adf.Linear(nb_states, hidden1, keep_variance_fn=self.keep_variance_fn)
        self.linear2 = adf.Linear(hidden1 + nb_actions, hidden2, keep_variance_fn=self.keep_variance_fn)
        self.linear3 = adf.Linear(hidden2, 1, keep_variance_fn=self.keep_variance_fn)

        self.ReLU = adf.ReLU(keep_variance_fn=self.keep_variance_fn)

        self.dropout = adf.Dropout(p=self.dropout_p, keep_variance_fn=self.keep_variance_fn)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.linear1.weight.data = fanin_init(self.linear1.weight.data.size())
        self.linear2.weight.data = fanin_init(self.linear2.weight.data.size())
        self.linear3.weight.data.uniform_(-init_w, init_w)

    def forward(self, state, state_noise, action, action_uncertainty):

        inputs_mean = state
        inputs_var = torch.zeros_like(inputs_mean) + self._noise_variance
        # inputs_var = state_noise
        inputs_var = inputs_var + 2

        inputs_mean_nor = (inputs_mean - torch.min(inputs_mean)) / (torch.max(inputs_mean)- torch.min(inputs_mean))

        tmp_input = inputs_mean_nor+0.1, inputs_var

        sampled_output_mean = []
        sampled_output_var = []

        with torch.no_grad():
            for _ in range(self.dropout_n):
                out = self.linear1(*tmp_input)
                out = self.ReLU(*out)
                out = self.dropout(*out)
                new_input = self.input_concat(out, action.cuda(), action_uncertainty.cuda())
                out = self.linear2(*new_input)
                out = self.ReLU(*out)
                out = self.dropout(*out)
                out = self.linear3(*out)

                sampled_output_mean.append(out[0])
                sampled_output_var.append(out[1])

        for i in range(len(sampled_output_mean)):
            sampled_output_mean[i] = sampled_output_mean[i].cpu().detach().numpy()
            sampled_output_var[i] = sampled_output_var[i].cpu().detach().numpy()


        np_sampled_output_mean = torch.mean(torch.tensor(sampled_output_mean), axis = 0).cuda()

        mean_var = []

        for i in range(len(sampled_output_mean)):
            tmp_mean_var = []
            for j in range(np_sampled_output_mean.shape[0]):
                tmp_mean_var.append( (np_sampled_output_mean[j].cpu().numpy() - sampled_output_mean[i][j]) ** 2)

            mean_var.append(tmp_mean_var)

        np_sampled_output_var = torch.mean(torch.tensor(sampled_output_var), axis=0) + torch.mean(torch.tensor(mean_var), axis=0)

        if np_sampled_output_var.shape[0] != 64:
            for i3 in range(len(np_sampled_output_var)):
                np_sampled_output_var[i3] = math.pow(10, np_sampled_output_var[i3])

        else:
            for i3 in range(np_sampled_output_var.shape[0]):
                for j3 in range(len(np_sampled_output_var[i3])):
                    np_sampled_output_var[i3][j3] = math.pow(10, np_sampled_output_var[i3][j3])

        return np_sampled_output_mean.cuda(), np_sampled_output_var.cuda()


    def input_concat(self, state, action_mean, action_var):
        return torch.cat((state[0], action_mean), 1), torch.cat((state[0], action_var), 1)



# class Actor(nn.Module):
#     def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(nb_states, hidden1)
#         self.fc2 = nn.Linear(hidden1, hidden2)
#         self.fc3 = nn.Linear(hidden2, nb_actions)
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()
#         self.init_weights(init_w)
#
#     def init_weights(self, init_w):
#         self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
#         self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
#         self.fc3.weight.data.uniform_(-init_w, init_w)
#
#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.relu(out)
#         out = self.fc3(out)
#         out = self.tanh(out)
#         return out
#
# class Critic(nn.Module):
#     def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
#         super(Critic, self).__init__()
#         self.nb_states = nb_states
#         self.fc1 = nn.Linear(nb_states, hidden1)
#
#         self.fc2 = nn.Linear(hidden1 + nb_actions, hidden2)
#         self.fc3 = nn.Linear(hidden2, 1)
#         self.relu = nn.ReLU()
#         self.init_weights(init_w)
#
#     def init_weights(self, init_w):
#         self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
#         self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
#         self.fc3.weight.data.uniform_(-init_w, init_w)
#
#     def forward(self, xs):
#         x, a = xs
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(torch.cat([out, a], 1))
#         out = self.relu(out)
#         out = self.fc3(out)
#         return out

