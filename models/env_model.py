import sys
sys.path.append('../')

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from networks.modelEnv import EnvModelNetwork
from params import Params as p
import minipacman.minipacman_utils as mpu

from scipy.stats import entropy
from scipy.special import entr


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EnvModel():

    def __init__(self, obs_space, num_pixels, num_rewards):
        #torch.manual_seed(0)
        self.obs = obs_space          # input
        #self.actions = actions  # output
        # self.seed = random.seed(seed)

        self.envNet = EnvModelNetwork(obs_space, num_pixels, num_rewards).to(device)
        self.optimizer = optim.Adam(self.envNet.parameters())

    # TODO: SAVE AND LOAD ALL NETWORKS
    def get_state_dict(self):
        return self.envNet.state_dict()

    def load_state_dict(self, fname, rpf=True):
        self.envNet.load_state_dict(torch.load(fname))

    ''' return imagined states and imagined rewards '''
    def imagine(self, x):
        observations, rewards = self.envNet(x)
        return observations, rewards

    def learn(self, imagined_state, target_state, imagined_reward, target_reward):
        #print('target reward ', tr.shape)
        #print('target state ', ts.shape)
        # Make targets into 1-D, each row is the target value
        #tr = target_reward.squeeze(1)
        #ts = target_state.view(-1)
        #print('target reward ', tr.shape)
        #print('target state ', ts.shape)

        criterion = nn.CrossEntropyLoss()#reduction='none')

        # Rew Loss
        # A loss PER head PER instance of the batch
        reward_loss = criterion(imagined_reward, target_reward)#.unsqueeze(1)

        # Obs loss
        #print(batch_img_states[head].shape, ts.shape)
        obs_loss = criterion(imagined_state, target_state)#.unsqueeze(1)

        # Backprop
        # loss = total_obs_loss + p.REW_WEIGHT * total_reward_loss
        loss = obs_loss + p.REW_WEIGHT * reward_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.envNet.parameters(), p.MAX_GRADNORM)
        self.optimizer.step()


        return loss, obs_loss, reward_loss