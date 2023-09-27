import sys
sys.path.append('../')
# from utils.plotting import plot_rew_time
from params import Params as p
from RH.rolling_horizon import RollingHorizon

# from Model.env_model import EnvModel
from minipacman.minipacman import MiniPacman
import minipacman.minipacman_utils as mpu
# from utils.misc import one_hot_encoding
from tensorboardX import SummaryWriter
import re
import numpy as np
import copy

import os
from os.path import join
from torchvision.utils import save_image
import torch


#from a2c import A2C
from models.env_model import EnvModel


def processImgToSave(image):
    # Transform 1-D 285 pixel vector from its categorical form into pixel-channel form (285 x 3)
    image = mpu.target_to_pix(image.cpu().numpy())
    # reshape into image dimensions (1 x 15 x 19 x 3)
    image = torch.FloatTensor(image.reshape(15, 19, 3)).unsqueeze(0)
    image = image.permute(0, 3, 1, 2)
    return image

def saveImg(state, dir, fname):
    state_image = torch.FloatTensor(state).unsqueeze(0)  # .permute(1, 2, 0).cpu().numpy()
    if not os.path.isdir(dir):
        os.makedirs(dir)
    save_image(state_image, join(dir + fname))


# Init Env
mode = "regular"
env = MiniPacman(mode, 1000)
state = env.reset()
#env.seed(0)

input_space = env.observation_space.shape
action_space = env.action_space.n

# Set environment model
env_model = EnvModel(input_space, mpu.num_pixels, mpu.num_rewards)
env_model.load_state_dict('worldModels/env_model_uni_a2c_50k-rf')

# Init Rolling Horizon agent
agentRH = RollingHorizon(input_space, action_space, env_model)



epiRewards = 0
rewards = []
steps = []
last_t = 0
episode = 0
saveImg(state, 'samples-rh+model/real/', join('ep' + str(episode) + 'init.png'))
for i in range(p.STEPS):

    # Pass current state and a perfect copy of the environment
    # and get an action sequence
    action_seq = agentRH.select_action(state, saveDir='samples-rh+model/simulated/')
    state, reward, done, _ = env.step(action_seq[0])
    epiRewards += reward

    #print(action_seq[0])

    if done:
        saveImg(state, 'samples-rh+model/real/', join('ep' + str(episode) + 'end.png'))
        state = env.reset()
        rewards.append(epiRewards)
        steps.append(i-last_t)
        epiRewards = 0
        print('Episode {} Steps {} Rewards {}'.format(episode, i-last_t, rewards[-1]))
        last_t = i
        episode += 1
        saveImg(state, 'samples-rh+model/real/', join('ep' + str(episode) + 'end.png'))
        agentRH.curr_rollout = None

        if episode > 9:
            break

print('Reward Average {} Std Dev {}'.format(np.mean(rewards), np.std(rewards)))
print('Step Average {} Std Dev {}'.format(np.mean(steps), np.std(steps)))