import sys
sys.path.append('../')

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from networks.modelEnvBootstrappedRPF import EnvModelBootstrappedNetwork, EnvModelRPF
from params import Params as p
import minipacman.minipacman_utils as mpu

from scipy.stats import entropy
from scipy.special import entr


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EnvModel():

    def __init__(self, obs_space, num_pixels, num_rewards, num_heads):
        #TODO:seed
        #torch.manual_seed(0)
        self.obs = obs_space          # input
        #self.actions = actions  # output
        # self.seed = random.seed(seed)

        self.envNet = EnvModelBootstrappedNetwork(obs_space, num_pixels, num_rewards, num_heads).to(device)
        self.priorNet = EnvModelBootstrappedNetwork(obs_space, num_pixels, num_rewards, num_heads).to(device)

        self.rpfNet = EnvModelRPF(self.envNet, self.priorNet, p.PRIOR_SCALE)

        self.optimizer = optim.Adam(self.rpfNet.parameters())

    # TODO: SAVE AND LOAD ALL NETWORKS
    def get_state_dict(self):
        return self.rpfNet.state_dict()

    def load_state_dict(self, fname, rpf=True):
        if rpf:
            self.rpfNet.load_state_dict(torch.load(fname))
        else:
            self.envNet.load_state_dict(torch.load(fname))

    ''' return imagined states and imagined rewards '''
    def imagine(self, x, head_idxs, rpf=True):
        #states, rewards = self.envNet(x, head_idxs)
        #return states, rewards
        observations, rewards = self.rpfNet(x, head_idxs)
        return observations, rewards

    def learn(self, ir, ts, tr, state, action, m, action_space, input_space):
        #print('target reward ', tr.shape)
        #print('target state ', ts.shape)
        pixPerState = ts.shape[1] # Get the number of pixels in each frame
        # Make targets into 1-D, each row is the target value
        tr = tr.squeeze(1)
        ts = ts.view(-1)
        #print('target reward ', tr.shape)
        #print('target state ', ts.shape)

        batch_size = state.size(0)  #num of elements in minibatch
        # Create a tensor: same dimensions as frame and for each action
        onehot_actions = torch.zeros(batch_size, action_space, *input_space[1:]).to(device)
        # And fill the action grid accordingly
        onehot_actions[range(batch_size), action] = 1
        # Concat states and actions
        state_action = torch.cat([state, onehot_actions], 1).to(device)

        criterion = nn.CrossEntropyLoss(reduction='none')
        reward_losses, obs_losses = [], []
        # Get (batch) predictions from each head
        # TODO: what is better? to generate them here, or to generate them online for all heads during trajectory and
        # and store them in the buffer?
        batch_img_states, batch_img_rewards = self.imagine(state_action, list(range(p.NUM_HEADS)))  # 10 torch.Size([32, 10]), 10 torch.Size([9120, 7])
        #batch_img_rewards = self.imagine(state_action, list(range(p.NUM_HEADS)))
        #print('env_model.py learn 1 ', len(batch_img_rewards))
        #print('env_model.py learn 2 ', batch_img_rewards[0].shape, tr.shape, batch_img_states[0].shape, ts.shape)
        #
        # print(m)

        for head in range(p.NUM_HEADS):
            used = torch.sum(m[:, head])    # Instances of the batch used by this head
            # Rew Loss
            # A loss PER head PER instance of the batch
            reward_loss = criterion(batch_img_rewards[head], tr).unsqueeze(1)

            #print(reward_loss)

            #print(batch_img_rewards[head].shape, tr.shape)
            #print('reward loss ', reward_loss.shape)
            reward_loss = torch.sum(reward_loss * m[:, head].unsqueeze(1) / p.NUM_HEADS) / used
            reward_losses.append(reward_loss.unsqueeze(0))

            # Obs loss
            #print(batch_img_states[head].shape, ts.shape)
            obs_loss = criterion(batch_img_states[head], ts).view(batch_size, -1)#.unsqueeze(1)

            #print(obs_loss)

            # now i need to get batch instances to anulate them with the masks
            #print('obs loss ', obs_loss.shape, m[:, head].unsqueeze(1).shape)
            # Get average (although we avg over only the number of heads used not the total heads)
            # NOTE: The loss is very big, (e.g. rew loss is 1.1598 vs obs loss 277.5) should we also avg over num of pixels?
            #obs_loss = torch.sum(obs_loss * m[:, head].unsqueeze(1) / p.NUM_HEADS) / used
            obs_loss = torch.sum((obs_loss * m[:, head].unsqueeze(1) / pixPerState) / p.NUM_HEADS) / used
            #print(reward_loss, obs_loss, obs_lossPxAvg, obs_loss.unsqueeze(0))
            obs_losses.append(obs_loss.unsqueeze(0))
            #print(reward_loss, obs_loss)

        #print('losses')
        #print('r losses ', reward_losses)
        #print(torch.cat(reward_losses))
        # Now sum each head's loss and get the average
        reward_losses = torch.cat(reward_losses)
        total_reward_loss = reward_losses.sum() / p.NUM_HEADS
        #print('o losses ', obs_losses)
        obs_losses = torch.cat(obs_losses)
        total_obs_loss = obs_losses.sum() / p.NUM_HEADS
        #print(total_reward_loss, total_obs_loss)

        # Backprop
        # loss = total_obs_loss + p.REW_WEIGHT * total_reward_loss
        loss = total_obs_loss + total_reward_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.rpfNet.parameters(), p.MAX_GRADNORM)
        self.optimizer.step()

        # We're going to compute entropies
        # First get predicted reward label (argmax) and pixel type
        # permute to get row=batch instance, col=head
        pred_rew = torch.stack(batch_img_rewards).argmax(2).permute(1,0).cpu().data.numpy()
        pred_pix = torch.stack(batch_img_states).argmax(2).permute(1,0).cpu().data.numpy()


        # This entropy is to track how the model evolves in training time

        # To compute actual entropy of reward label distribution
        #TODO: i could also just compute a pseudo-entropy based on the head voting
        # Because depending on the heads is also my measure of entropy
        # it's not something that is invariant
        # I can also just get max entropies or something like that
        entropyRew = self.entropyRow(pred_rew)  # get entropy of every instance of a predicted reward (so size is minibatch size)
        entropyPix = self.entropyRow(pred_pix)

        return loss, total_obs_loss, obs_losses, total_reward_loss, reward_losses, entropyPix.mean(), entropyRew.mean()


    def entropyRow(self, arr):
        # This is to create the upper boundaries of the bins
        N = arr.max() + 1
        # The ids is to assign unique ranges by row, since we're interested in computing entropy per isolated row
        id = arr + (N * np.arange(arr.shape[0]))[:, None]
        # This just simply does the counts per elements row wise
        counts = np.bincount(id.ravel(), minlength=N * arr.shape[0]).reshape(-1, N)
        pr = counts / counts.sum(axis=1, keepdims=True)
        # entr(pr) = -p log p THUS entr(pr).sum() = - Σ p_i log p_i
        entropies = entr(pr).sum(axis=1)
        return entropies


# class EnvModelNonProb():
#
#     def __init__(self, obs_space, num_pixels, num_rewards):
#         #TODO:seed
#         #torch.manual_seed(0)
#         self.obs = obs_space          # input
#         #self.actions = actions  # output
#         # self.seed = random.seed(seed)
#
#         self.envNet = EnvModelNetwork(obs_space, num_pixels, num_rewards).to(device)
#         self.optimizer = optim.Adam(self.envNet.parameters())
#
#     # TODO: SAVE AND LOAD ALL NETWORKS
#     def get_state_dict(self):
#         return self.envNet.state_dict()
#
#     def load_state_dict(self, fname, rpf=True):
#         self.envNet.load_state_dict(torch.load(fname))
#
#     ''' return imagined states and imagined rewards '''
#     def imagine(self, x):
#         observations, rewards = self.envNet(x)
#         return observations, rewards
#
#     def learn(self, imagined_state, target_state, imagined_reward, target_reward):
#         #print('target reward ', tr.shape)
#         #print('target state ', ts.shape)
#         # Make targets into 1-D, each row is the target value
#         #tr = target_reward.squeeze(1)
#         #ts = target_state.view(-1)
#         #print('target reward ', tr.shape)
#         #print('target state ', ts.shape)
#
#         criterion = nn.CrossEntropyLoss()#reduction='none')
#
#         # Rew Loss
#         # A loss PER head PER instance of the batch
#         reward_loss = criterion(imagined_reward, target_reward)#.unsqueeze(1)
#
#         # Obs loss
#         #print(batch_img_states[head].shape, ts.shape)
#         obs_loss = criterion(imagined_state, target_state)#.unsqueeze(1)
#
#         # Backprop
#         # loss = total_obs_loss + p.REW_WEIGHT * total_reward_loss
#         loss = obs_loss + p.REW_WEIGHT * reward_loss
#         self.optimizer.zero_grad()
#         loss.backward()
#         nn.utils.clip_grad_norm_(self.envNet.parameters(), p.MAX_GRADNORM)
#         self.optimizer.step()
#
#
#         return loss, obs_loss, reward_loss