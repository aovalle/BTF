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
simulator = MiniPacman(mode, 1000)
state = env.reset()
#env.seed(0)

input_space = env.observation_space.shape
action_space = env.action_space.n

# Init Rolling Horizon agent
agentRH = RollingHorizon(input_space, action_space)



epiRewards = 0
rewards = []
last_t = 0
episode = 0
saveImg(state, 'samples-rh/real/', join('ep' + str(episode) + 'init.png'))
for i in range(p.STEPS):

    # Get a copy of the world state
    cloned_state = copy.deepcopy(env.env.world_state)
    # Set simulator to that state

    simulator = copy.deepcopy(env)

    simulator.env.world_state = cloned_state
    # comparison_state = copy.deepcopy(env.env.world_state)

    # Pass current state and a perfect copy of the environment
    # and get an action sequence
    action_seq = agentRH.select_action(state, simulator, cloned_state)




    # Get a copy of the world state
    cloned_state = copy.deepcopy(env.env.world_state)
    # Set simulator to that state
    simulator = copy.deepcopy(env)
    simulator.env.world_state = cloned_state
    agentRH.testBestRoll(state, simulator, cloned_state, action_seq[1])
    print(action_seq)
    print('GHOS REAL ', env.get_ghost_pos())
    print('PAC REAL ', env.get_ghost_pos())
    for st in range(len(action_seq[1])):
        state, reward, done, _ = env.step(action_seq[1][st])
        print('GHOS REAL ', env.get_ghost_pos())
        print('PAC REAL ', env.get_ghost_pos())
    exit()






    state, reward, done, _ = env.step(action_seq[0])
    epiRewards += reward

    #print(action_seq[0])

    if done:
        saveImg(state, 'samples-rh/real/', join('ep' + str(episode) + 'end.png'))
        state = env.reset()
        rewards.append(epiRewards)
        epiRewards = 0
        print('Episode {} Steps {} Rewards {}'.format(episode, i-last_t, rewards[-1]))
        last_t = i
        episode += 1
        saveImg(state, 'samples-rh/real/', join('ep' + str(episode) + 'end.png'))
        agentRH.curr_rollout = None

        if episode > 10:
            break

print('Average {} Std Dev {}'.format(np.mean(rewards), np.std(rewards)))