#!/usr/bin/env python3 

import numpy as np
import argparse
from copy import deepcopy
import torch
import gym
import os

from normalized_env import NormalizedEnv
from evaluator import Evaluator
from ddpg import DDPG_with_dropout
from util import *
from datetime import datetime

# gym.undo_logger_setup()


def train(num_iterations, agent, env,  evaluate, validate_steps, model_output, train_with_dropout, max_episode_length=None, debug=False):

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    obs = None
    while step < num_iterations:
        # reset if it is the start of episode
        if obs is None:
            obs = deepcopy(env.reset())
            agent.reset(obs)

        # agent pick action ...
        if step <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action_with_dropout(obs, train_with_dropout)

        # env response with obs, reward, terminate_info
        next_obs, reward, done, info = env.step(action)

        next_obs = deepcopy(next_obs)
        if max_episode_length and episode_steps >= max_episode_length - 1:
            done = True

        # agent observe and update policy
        agent.observe(reward, next_obs, done)
        if step > args.warmup:
            agent.update_policy()
        
        # [optional] evaluate
        if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
            policy = lambda x: agent.select_action_with_dropout(x, train_with_dropout, decay_epsilon=False)
            validate_reward = evaluate(env, policy, debug=False, visualize=False)
            if debug:
                prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))

        # [optional] save intermideate model
        if step % int(num_iterations/3) == 0:
            agent.save_model(model_output)

        # update 
        step += 1
        episode_steps += 1
        episode_reward += reward
        obs = deepcopy(next_obs)

        if done:  # end of episode
            if debug:
                prGreen('#{}: episode_reward:{} steps:{}'.format(episode, episode_reward, step))

            agent.memory.append(obs, agent.select_action_with_dropout(obs), 0., False)

            # reset
            obs = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1


def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action_with_dropout(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        if debug:
            prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))


if __name__ == "__main__":

    now = datetime.now()
    now_date = str(now.year)[-2:] + str(now.month).zfill(2) + str(now.day).zfill(2)
    now_time = str(now.hour).zfill(2) + str(now.minute).zfill(2)

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env', default='MountainCarContinuous-v0', type=str, help='open-ai gym environment')
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=100, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--train_iter', default=2000000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    parser.add_argument('--dropout_p', default=0.2, type=float, help='Bernoulli possibility p for dropout')
    parser.add_argument('--dropout_n', default=3, type=int, help='Monte Carlo Dropout Sampling Number')
    parser.add_argument('--train_with_dropout', action='store_true', help='is_training')

    args = parser.parse_args()
    args.output = args.output + '/' + now_date + '/' + now_time
    model_output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)

    np.savetxt(args.output + '/' + str(args.dropout_n) + '_' + str(args.dropout_p) + '_' + str(args.train_with_dropout) + '.txt', np.array([]), fmt='%4.6f', delimiter=' ')

    env = NormalizedEnv(gym.make(args.env))

    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]

    agent = DDPG_with_dropout(nb_states, nb_actions, args.dropout_n, args.dropout_p, args)

    evaluate = Evaluator(args.validate_episodes, args.validate_steps, model_output, max_episode_length=args.max_episode_length)

    if args.mode == 'train':
        train(args.train_iter, agent, env, evaluate, args.validate_steps, model_output, args.train_with_dropout, max_episode_length=args.max_episode_length, debug=args.debug)

    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.resume, visualize=True, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
