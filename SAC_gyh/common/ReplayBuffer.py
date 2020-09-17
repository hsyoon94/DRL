import numpy as np
from random import *



class Buffer():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.buffer = []

    def add(self, observation, action, reward, next_observation, done):
        self.buffer.insert(0, [np.array(observation), np.array(action), np.array(reward), np.array(next_observation), np.array(done)])

        if len(self.buffer) > self.batch_size:
            self.buffer.pop()

    def sample(self):
        index = randint(0, len(self.buffer))
        return self.buffer[index]
