import torch
import numpy as np
import os

model_path = '/home/hsyoon/job/DRL/DDPG/output/200623/150243/MountainCarContinuous-v0-run1/'

for root, subdirs, files in os.walk(model_path):
    for subdir in subdirs:
        model = torch.load(root + subdir + '/actor.pkl')
        # print(model)
        for i in model["fc2.weight"][0]:
            with open(root + subdir + "/actor.txt", "a") as myfile:
                myfile.write(str(i.cpu().numpy()) + '\n')

        print(str(subdir) + "saved!")





