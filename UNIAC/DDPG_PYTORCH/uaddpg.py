
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import UAActor, UACritic
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *
import os
import os
from contrib import adf
import math

# from ipdb import set_trace as debug

def keep_variance(x, min_variance):
    return x + min_variance


def one_hot_pred_from_label(y_pred, labels):
    y_true = torch.zeros_like(y_pred)
    ones = torch.ones_like(y_pred)
    indexes = [l for l in labels]

    print("y_pred", y_pred)
    print("labels", labels)

    print(labels.size(0))
    print("indexex", indexes)
    y_true[torch.arange(labels.size(0)), indexes] = ones[torch.arange(labels.size(0)), indexes]

    return y_true

class SoftmaxHeteroscedasticLoss(torch.nn.Module):
    def __init__(self):
        super(SoftmaxHeteroscedasticLoss, self).__init__()

        min_variance = 1e-3

        keep_variance_fn = lambda x: keep_variance(x, min_variance=min_variance)
        self.adf_softmax = adf.Softmax(dim=1, keep_variance_fn=keep_variance_fn)

    def forward(self, outputs, targets, eps=1e-5):
        mean, var = self.adf_softmax(*outputs)
        # targets = one_hot_pred_from_label(mean, targets)

        precision = 1 / (var + eps)
        return torch.mean(0.5 * precision * (targets - mean) ** 2 + 0.5 * torch.log(var + eps))


class UADDPG(object):
    def __init__(self, nb_states, nb_actions, args):
        print("UADDPG!!!!!!!!!!!!!!!!!!!!!!!!!")
        if args.seed > 0:
            self.seed(args.seed)

        self.episode = 0
        self.nb_states = nb_states
        self.nb_actions = nb_actions

        # Create Actor and Critic Network
        net_cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
            'init_w': args.init_w,
        }
        self.criterion = nn.MSELoss()

        self.actor = UAActor(self.nb_states, self.nb_actions, False, ** net_cfg)
        self.actor_target = UAActor(self.nb_states, self.nb_actions, True, ** net_cfg)

        self.actor_optim = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = UACritic(self.nb_states, self.nb_actions, False, **net_cfg)
        self.critic_target = UACritic(self.nb_states, self.nb_actions, True, **net_cfg)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.rate)

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

        # Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu,
                                                       sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        #
        self.epsilon = 1.0
        self.s_t = None  # Most recent state
        self.a_t_mean = None  # Most recent action
        self.a_t_var = None
        self.is_training = True

        #
        if USE_CUDA: self.cuda()

    def update_policy(self):
        # Sample batch
        state_batch, action_mean_batch, action_var_batch, reward_batch, next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q

        with torch.no_grad():
            next_q_values = self.critic_target([to_tensor(next_state_batch, volatile=True), self.actor_target(to_tensor(next_state_batch, volatile=True))[0], self.actor_target(to_tensor(next_state_batch, volatile=True))[1]])

        target_q_batch_mean = to_tensor(reward_batch) + self.discount * to_tensor(terminal_batch.astype(np.float)) * next_q_values[0]
        target_q_batch_var = to_tensor(reward_batch) + self.discount * to_tensor(terminal_batch.astype(np.float)) * next_q_values[1]

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([to_tensor(state_batch), to_tensor(action_mean_batch), to_tensor(action_var_batch)])
        value_loss = self.criterion(q_batch[0], target_q_batch_mean) + self.criterion(q_batch[1], target_q_batch_var)

        # print("action_mean_batch", action_mean_batch)
        # print("action_var_batch", action_var_batch)
        # print("q_batch[0]", q_batch[0])
        # print("target_q_batch_mean", target_q_batch_mean)
        # print("q_batch[1]", q_batch[1])
        # print("target_q_batch_var", target_q_batch_var)

        # with torch.no_grad():
        #     print("cri1", self.criterion(q_batch[0], target_q_batch_mean))
        #     print("cri2", self.criterion(q_batch[1], target_q_batch_var))
        #
        # print("value_loss", value_loss)

        # if torch.isfinite(value_loss) is False:
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss_mean = -self.critic([to_tensor(state_batch), self.actor(to_tensor(state_batch))[0], self.actor(to_tensor(state_batch))[1]])[0]
        policy_loss_var = -self.critic([to_tensor(state_batch), self.actor(to_tensor(state_batch))[0], self.actor(to_tensor(state_batch))[1]])[1]

        policy_loss = policy_loss_mean.mean() + policy_loss_var.mean()

        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t_mean, self.a_t_var, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        action_mean = np.random.uniform(-1., 1., self.nb_actions)
        action_var = np.random.uniform(-2., 2., self.nb_actions)
        self.a_t_mean = action_mean
        self.a_t_var = action_var
        return action_mean

    def select_action(self, s_t, decay_epsilon=True):

        action_mean = to_numpy(self.actor(to_tensor(np.array([s_t])))[0]).squeeze(0)
        action_var = to_numpy(self.actor(to_tensor(np.array([s_t])))[1]).squeeze(0)

        action_mean += self.is_training * max(self.epsilon, 0) * self.random_process.sample()

        action_mean = np.clip(action_mean, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon

        self.a_t_mean = action_mean
        self.a_t_var = action_var

        return action_mean

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )

    def save_model(self, output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def seed(self, s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
