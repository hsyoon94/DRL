import numpy as np
import os

import torch
import torch.nn as nn
from torch.optim import Adam

from model import ActorWithDropout, Critic
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *

# from ipdb import set_trace as debug

criterion = nn.MSELoss()

class DDPG_with_dropout(object):
    def __init__(self, nb_states, nb_actions, sample_n, dropout, args):
        
        if args.seed > 0:
            self.seed(args.seed)

        self.print_var_count = 0
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.dropout = dropout
        self.sample_n = sample_n
        self.action_std = np.array([])
        self.save_dir = args.output

        # Create Actor and Critic Network
        net_cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
            'init_w': args.init_w
        }

        self.DO_actor = ActorWithDropout(self.nb_states, self.nb_actions, sample_n=self.sample_n, dropout=self.dropout, **net_cfg)
        self.DO_actor_target = ActorWithDropout(self.nb_states, self.nb_actions, sample_n=self.sample_n, dropout=self.dropout, **net_cfg)
        self.DO_actor_optim = Adam(self.DO_actor.parameters(), lr=args.prate)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.rate)

        hard_update(self.DO_actor_target, self.DO_actor)
        hard_update(self.critic_target, self.critic)
        
        # Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        # 
        self.epsilon = 1.0
        self.s_t = None  # Most recent state
        self.a_t = None  # Most recent action
        self.is_training = True

        # 
        if USE_CUDA:
            self.cuda()

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target([to_tensor(next_state_batch, volatile=True), self.DO_actor_target(to_tensor(next_state_batch, volatile=True)),])

        target_q_batch = to_tensor(reward_batch) + self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([to_tensor(state_batch), to_tensor(action_batch)])
        
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.DO_actor.zero_grad()

        policy_loss = -self.critic([to_tensor(state_batch), self.DO_actor(to_tensor(state_batch))])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.DO_actor_optim.step()

        # Target update
        soft_update(self.DO_actor_target, self.DO_actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        # self.DO_actor.eval()
        # self.DO_actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.DO_actor.cuda()
        self.DO_actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1., 1., self.nb_actions)
        self.a_t = action
        return action

    def select_action_with_dropout(self, s_t, train_with_dropout=False, decay_epsilon=True):
        dropout_actions = np.array([])
        # for i in range(self.sample_n):
        #     action = to_numpy(self.DO_actor.forward_with_dropout(to_tensor(np.array([s_t])))).squeeze(0)
        #     dropout_actions = np.append(dropout_actions, [action])

        # plt action : without dropout
        if train_with_dropout:
            plt_action = to_numpy(self.DO_actor.forward_with_dropout(to_tensor(np.array([s_t])))).squeeze(0)

        else:
            plt_action = to_numpy(self.DO_actor(to_tensor(np.array([s_t])))).squeeze(0)
            plt_action += self.is_training * max(self.epsilon, 0) * self.random_process.sample()

        # if self.print_var_count % 9999 == 0:
        #     # print("dropout actions", dropout_actions)
        #     # print("dropout actions mean", np.mean(dropout_actions))
        #
        #     print("dropout actions std", np.std(dropout_actions))
        #     # if os.path.isdir(self.save_dir + '/std/'):
        #     #     os.mkdir(self.save_dir + '/std/')
        #     self.action_std = np.append(self.action_std, [np.std(dropout_actions)])
        #     np.savetxt(self.save_dir + '/std.txt', self.action_std, fmt='%4.6f', delimiter=' ')

            # print(plt_action)

        # if s_t[0] == -0.5 and s_t[1] == 0:
        #     print("initial dropout actions std", np.std(dropout_actions), "          ", self.is_training)

        # plr_avg_action = np.float32(np.mean(dropout_actions))

        self.print_var_count = self.print_var_count + 1

        # Delete gaussian random noise
        # action += self.is_training * max(self.epsilon, 0) * self.random_process.sample()
        plt_action = np.clip(plt_action, -1., 1.)
        # plr_avg_action = np.clip(plr_avg_action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon

        self.a_t = plt_action

        return plt_action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()


    def load_weights(self, output):
        if output is None:
            return

        self.DO_actor.load_state_dict(torch.load('{}/actor.pkl'.format(output)))
        self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(output)))


    def save_model(self,output):
        torch.save(self.DO_actor.state_dict(), '{}/actor.pkl'.format(output))
        torch.save(self.critic.state_dict(), '{}/critic.pkl'.format(output))


    def seed(self, s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)