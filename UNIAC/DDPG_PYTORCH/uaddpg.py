
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

from datetime import datetime

now = datetime.now()
now_date = str(now.year)[-2:] + str(now.month).zfill(2) + str(now.day).zfill(2)
now_time = str(now.hour).zfill(2) + str(now.minute).zfill(2)

# from ipdb import set_trace as debug

def keep_variance(x, min_variance):
    # return x + min_variance
    return min_variance

def KLDLoss(mean1, var1, mean2, var2):
    kld1 = torch.log((torch.sqrt(torch.abs(var2) + 1e-16))/(torch.sqrt(torch.abs(var1)) + 1e-16)) + (torch.abs(var1) + (mean1 - mean2)**2)/(2 * torch.abs(var2) + 1e-16)
    kld2 = torch.log((torch.sqrt(torch.abs(var1) + 1e-16))/(torch.sqrt(torch.abs(var2)) + 1e-16)) + (torch.abs(var2) + (mean2 - mean1)**2)/(2 * torch.abs(var1) + 1e-16)

    error = 0

    for value1 in kld1:
        if math.isnan(value1) is False:
            error = error + value1 * value1

    for value2 in kld2:
        if math.isnan(value2) is False:
            error = error + value2 * value2

    error.requires_grad = True
    error = torch.sqrt(error) / 100
    return error

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
    def __init__(self, nb_states, nb_actions, now_date, now_time, args):
        print("UADDPG!!!!!!!!!!!!!!!!!!!!!!!!!")
        if args.seed > 0:
            self.seed(args.seed)

        self.total_training_step = 1
        self.episode = 0
        self.nb_states = nb_states
        self.nb_actions = nb_actions

        # Create Actor and Critic Network
        net_cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
            'init_w': args.init_w
        }
        # self.criterion = nn.MSELoss()
        self.critic_case = 'stochastic'
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
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        self.epsilon = 1.0
        self.s_t = None  # Most recent state
        self.s_t_noise = None  # Most recent state
        self.a_t_mean = None  # Most recent action
        self.a_t_var = None
        self.is_training = True

        if torch.cuda.is_available():
            self.cuda()

        self.now_date = now_date
        self.now_time = now_time

        if os.path.exists('/mnt/sda2/DRL/UNIAC/model_' + self.now_date + '_' + self.now_time + '/') is False:
            os.mkdir('/mnt/sda2/DRL/UNIAC/model_' + self.now_date + '_' + self.now_time+ '/')

    def update_policy(self):
        # print("Policy update starts...")
        # Sample batch
        state_batch, state_noise_batch, action_mean_batch, action_var_batch, reward_batch, next_state_batch, next_state_noise_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q

        with torch.no_grad():
            action_mean, action_var = self.actor_target(to_tensor(next_state_batch, volatile=True), to_tensor(next_state_noise_batch, volatile=True))
            next_q_values = self.critic_target(to_tensor(next_state_batch, volatile=True), to_tensor(next_state_noise_batch, volatile=True), action_mean, action_var)

        target_q_batch_mean = to_tensor(reward_batch) + self.discount * to_tensor(terminal_batch.astype(np.float)) * next_q_values[0]
        target_q_batch_var = to_tensor(reward_batch) + self.discount * to_tensor(terminal_batch.astype(np.float)) * next_q_values[1]

        # Critic update
        self.critic.zero_grad()

        # case 1 : Stochastic error (KL Divergence on both distribution)
        if self.critic_case == 'stochastic':

            q_batch = self.critic(to_tensor(state_batch), to_tensor(state_noise_batch), to_tensor(action_mean_batch), to_tensor(action_var_batch))
            value_loss = KLDLoss(q_batch[0], q_batch[1], target_q_batch_mean, target_q_batch_var)

        # case 2 : Deterministic error (MSE error)
        else:
            q_batch_sample = []
            target_q_batch_sample = []
            q_batch = self.critic([to_tensor(state_batch), action_mean_batch, action_var_batch])
            for q_index in range(action_var_batch.shape[0]):
                q_batch_sample[q_index] = q_batch[0][q_index] - q_batch[1][q_index]
                target_q_batch_sample[q_index] = target_q_batch_mean[q_index] - target_q_match_var[q_index]
            value_loss = nn.MSE(q_batch_sample, target_q_batch_sample)

        value_loss.backward()

        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        action_mean, action_var = self.actor(to_tensor(state_batch), to_tensor(state_noise_batch))

        policy_loss_mean, policy_loss_var = self.critic(to_tensor(state_batch), to_tensor(state_noise_batch), action_mean, action_var)
        # policy_loss_mean = -policy_loss_mean

        if self.critic_case == 'stochastic':
            # policy_loss = policy_loss_mean.mean() + policy_loss_var.mean()
            policy_loss = policy_loss_mean.mean()
        else:
            policy_loss = (policy_loss_mean - policy_loss_var).mean()

        policy_loss.requires_grad = True
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        # print("Policy update ends...")

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

    def observe(self, r_t, s_next_mean, s_next_var, done):
        if self.is_training:
            self.memory.append(self.s_t, self.s_t_noise, self.a_t_mean, self.a_t_var, r_t, done)
            self.s_t = s_next_mean
            self.s_t_noise = s_next_var

    def random_action(self):
        action_mean = np.random.uniform(-1., 1., self.nb_actions)
        action_var = np.random.uniform(-2., 2., self.nb_actions)
        self.a_t_mean = action_mean
        self.a_t_var = action_var
        return action_mean

    def select_action(self, s_t, s_t_noise, decay_epsilon=True):
        action_mean, action_var = self.actor(to_tensor(np.array([s_t])), to_tensor(np.array([s_t_noise])))

        action_noise = []

        # amplification = 10000 - self.total_training_step / 100
        # if amplification < 1:
        #     amplification = 1
        amplification = 1

        for index in range(action_mean.shape[0]):
            action_noise.append(np.random.normal(0, action_var.cpu()[index] * amplification, 1))

        # action_mean += self.is_training * max(self.epsilon, 0) * self.random_process.sample()

        action_sample = action_mean + max(self.epsilon, 0) * torch.tensor(np.array(action_noise).squeeze()).cuda()


        # print("action_mean", action_mean)
        # print("action_noise", action_noise)
        # print("action_sample", action_sample)

        # action_sample = np.clip(action_sample, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon

        self.a_t_mean = action_mean.cpu().numpy()
        self.a_t_var = action_var.cpu().numpy()
        self.total_training_step = self.total_training_step + 1

        action_mean_file = open('/mnt/sda2/DRL/UNIAC/model_' + self.now_date + '_' + self.now_time + '/action_mean.txt', 'a')
        action_var_file = open('/mnt/sda2/DRL/UNIAC/model_' + self.now_date + '_' + self.now_time + '/action_var.txt', 'a')
        action_noise_file = open('/mnt/sda2/DRL/UNIAC/model_' + self.now_date + '_' + self.now_time + '/action_noise.txt', 'a')
        action_sample_file = open('/mnt/sda2/DRL/UNIAC/model_' + self.now_date + '_' + self.now_time + '/action_sample.txt', 'a')

        action_mean_file.write(str(action_mean) + '\n')
        action_var_file.write(str(action_var) + '\n')
        action_noise_file.write(str(action_noise) + '\n')
        action_sample_file.write(str(action_sample) + '\n')

        action_mean_file.close()
        action_var_file.close()
        action_noise_file.close()
        action_sample_file.close()

        return action_sample.cpu().numpy()
        # return np.clip(action_sample.cpu().numpy(), -1.0, 1.0)

    def reset(self, obs, obs_noise):
        self.s_t = obs
        self.s_t_noise = obs_noise
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
