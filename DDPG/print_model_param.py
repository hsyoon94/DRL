import torch
import numpy as np
import os

model_path = '/home/hsyoon/job/DRL/DDPG/output/200624/142357/MountainCarContinuous-v0-run1/'

for root, subdirs, files in os.walk(model_path):
    for subdir in subdirs:
        model = torch.load(root + subdir + '/actor.pkl')

        for i in model["fc1.weight"]:
            with open(root + subdir + "/actor_fc1_weight[0].txt", "a") as myfile:
                myfile.write(str("%0.32F" % i[0].cpu().numpy()) + '\n')
            with open(root + subdir + "/actor_fc1_weight[1].txt", "a") as myfile:
                myfile.write(str("%0.32F" % i[1].cpu().numpy()) + '\n')

        for i in model["fc2.weight"][0]:
            with open(root + subdir + "/actor_fc2_weight[0].txt", "a") as myfile:
                myfile.write(str("%0.32F" % i.cpu().numpy()) + '\n')

        for i in model["fc3.weight"][0]:
            with open(root + subdir + "/actor_fc3_weight[0].txt", "a") as myfile:
                myfile.write(str("%0.32F" % i.cpu().numpy()) + '\n')

        print(str(subdir) + " saved!")