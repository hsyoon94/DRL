
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import (Actor, Critic, UAActor, UACritic)
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *
import os
import os

# from ipdb import set_trace as debug

criterion = nn.MSELoss()


class DDPG(object):
    def __init__(self, nb_states, nb_actions, args):

        if args.seed > 0:
            self.seed(args.seed)

        self.episode = 0
        self.nb_states = nb_states
        self.nb_actions = nb_actions

        # Create Actor and Critic Network
        net_cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
            'init_w': args.init_w
        }
        self.actor = UAActor(self.nb_states, self.nb_actions, **net_cfg)

        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
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
        self.a_t = None  # Most recent action
        self.is_training = True

        #
        if USE_CUDA: self.cuda()

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q

        # print("input x in xs", to_tensor(next_state_batch, volatile=True))
        # print("input s in xs", self.actor_target(to_tensor(next_state_batch, volatile=True)))

        next_q_values = self.critic_target([
            to_tensor(next_state_batch, volatile=True), self.actor_target(to_tensor(next_state_batch, volatile=True)),
        ])
        next_q_values.volatile = False

        target_q_batch = to_tensor(reward_batch) + \
                         self.discount * to_tensor(terminal_batch.astype(np.float)) * next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([to_tensor(state_batch), to_tensor(action_batch)])

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch))
        ])

        policy_loss = policy_loss.mean()
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
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1., 1., self.nb_actions)
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):

        # action = to_numpy(
        #     self.actor(to_tensor(np.array([s_t])))
        # ).squeeze(0)

        action = to_numpy(
            self.actor(to_tensor(np.array([s_t])))[0]
        ).squeeze(0)

        action += self.is_training * max(self.epsilon, 0) * self.random_process.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon

        self.a_t = action
        return action

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

class UADDPG(object):
    def __init__(self, nb_states, nb_actions, args):
        
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions = nb_actions

        self.epistemic_actor = args.epistemic_actor # true / false
        self.epistemic_critic = args.epistemic_critic # true / false

        self.aleatoric_actor = args.aleatoric_actor # true / false
        self.aleatoric_critic = args.aleatoric_critic # true / false

        self.dropout_n_actor = args.dropout_n_actor
        self.dropout_n_critic = args.dropout_n_critic

        self.dropout_p_actor = args.dropout_p_actor
        self.dropout_p_critic = args.dropout_p_critic

        self.print_var_count = 0
        self.action_std = np.array([])
        self.save_dir = args.output
        self.episode = 0

        # self.save_file = open(self.save_dir + '/std.txt', "a")


        # Create Actor and Critic Network
        net_cfg_actor = {
            'dropout_n' : args.dropout_n_actor,
            'dropout_p' : args.dropout_p_actor,
            'hidden1' : args.hidden1,
            'hidden2' : args.hidden2,
            'init_w' : args.init_w
        }

        net_cfg_critic = {
            'dropout_n' : args.dropout_n_actor,
            'dropout_p' : args.dropout_p_critic,
            'hidden1' : args.hidden1,
            'hidden2' : args.hidden2,
            'init_w' : args.init_w
        }

        self.actor = UAActor(self.nb_states, self.nb_actions, **net_cfg_actor)
        self.actor_target = UAActor(self.nb_states, self.nb_actions, **net_cfg_actor)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = UACritic(self.nb_states, self.nb_actions, **net_cfg_critic)
        self.critic_target = UACritic(self.nb_states, self.nb_actions, **net_cfg_critic)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.rate)

        hard_update(self.actor_target, self.actor)
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
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

        # 
        if USE_CUDA:
            self.cuda()

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        # TODO : Also apply epistemic and aleatoric uncertainty to both actor and critic target network
        next_q_values = self.critic_target([to_tensor(next_state_batch, volatile=True),self.actor_target(to_tensor(next_state_batch, volatile=True)),])
        target_q_batch = to_tensor(reward_batch) + self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values

        #########################
        #  Critic update
        #########################
        self.critic.zero_grad()

        # TODO : Add epistemic uncertainty for critic network
        q_batch = self.critic([ to_tensor(state_batch), to_tensor(action_batch)])

        # TODO : Add aleatoric uncertainty term from aleatoric uncertainty output of critic network (Add uncertainty term in criterion)
        value_loss = criterion(q_batch, target_q_batch)

        value_loss.backward()
        self.critic_optim.step()

        #########################
        #  Actor update
        #########################
        self.actor.zero_grad()

        # policy loss
        # TODO : Add epistemic certainty term from aleatoric certainty output of policy network
        policy_loss = -self.critic([to_tensor(state_batch), self.actor(to_tensor(state_batch))])
        policy_loss = policy_loss.mean()
        # policy_loss = policy_loss.mean() + actor_certainty

        policy_loss.backward()
        self.actor_optim.step()

        #########################
        #  Target soft update
        #########################
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
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1.,1.,self.nb_actions)
        self.a_t = action
        return action

    # def select_action(self, s_t, decay_epsilon=True):
    #     action = to_numpy(self.actor(to_tensor(np.array([s_t])))).squeeze(0)
    #     action += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
    #
    #     if decay_epsilon:
    #         self.epsilon -= self.depsilon
    #
    #     self.a_t = action
    #     return action

    def select_action_with_dropout(self, s_t, decay_epsilon=True):
        dropout_actions = np.array([])

        with torch.no_grad():
            for _ in range(self.dropout_n):
                action = to_numpy(self.actor.forward_with_dropout(to_tensor(np.array([s_t])))).squeeze(0)
                dropout_actions = np.append(dropout_actions, [action])

        if self.train_with_dropout:
            plt_action = to_numpy(self.actor.forward_with_dropout(to_tensor(np.array([s_t])))).squeeze(0)
            plt_action += self.is_training * max(self.epsilon, 0) * self.random_process.sample()

        else:
            plt_action = to_numpy(self.actor(to_tensor(np.array([s_t])))).squeeze(0)
            plt_action += self.is_training * max(self.epsilon, 0) * self.random_process.sample()

        """
        UNFIXED RESET POINT for Mujoco
        """
        if self.print_var_count != 0 and (self.print_var_count + 1) % 999 == 0:
            # self.action_std = np.append(self.action_std, [np.std(dropout_actions)])

            with open(self.save_dir + "/std.txt", "a") as myfile:
                myfile.write(str(np.std(dropout_actions))+'\n')
            with open(self.save_dir + "/mean.txt", "a") as myfile:
                myfile.write(str(np.mean(dropout_actions))+'\n')


        if self.print_var_count % (1000*5) == 0:
            print("dropout actions std", np.std(dropout_actions), "            ", "dir : ", str(self.save_dir))

        """
        FIXED RESET POINT for MCC
        """
        # if s_t[0] == -0.5 and s_t[1] == 0:
        #     # print("fixed dropout actions std", np.std(dropout_actions), "            ", "dir : ", str(self.save_dir))
        #     self.action_std = np.append(self.action_std, [np.std(dropout_actions)])
        #     # np.savetxt(self.save_dir + '/std.txt', self.action_std, fmt='%4.10f', delimiter=' ')
        #     with open(self.save_dir + "/std.txt", "a") as myfile:
        #         myfile.write(str(np.std(dropout_actions))+'\n')
        #     with open(self.save_dir + "/mean.txt", "a") as myfile:
        #         myfile.write(str(np.mean(dropout_actions))+'\n')

        if not (os.path.isdir(self.save_dir + "/episode/" + str(self.episode))):
            os.makedirs(os.path.join(self.save_dir + "/episode/" + str(self.episode)))

        self.action_std = np.append(self.action_std, [np.std(dropout_actions)])
        with open(self.save_dir + "/episode/" + str(self.episode) + "/std.txt", "a") as myfile:
            myfile.write(str(np.std(dropout_actions)) + '\n')

        with open(self.save_dir + "/episode/" + str(self.episode) + "/mean.txt", "a") as myfile:
            myfile.write(str(np.mean(dropout_actions)) + '\n')

        self.print_var_count = self.print_var_count + 1

        if decay_epsilon:
            self.epsilon -= self.depsilon


        self.a_t = plt_action

        return plt_action

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
