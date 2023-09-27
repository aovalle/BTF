from modelA2C import ACNetwork
from params import Params as p
from minipacman.minipacman import MiniPacman

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class A2C():

    def __init__(self, obs, actions):
        torch.manual_seed(0)
        self.obs = obs          # input
        self.actions = actions  # output
        # self.seed = random.seed(seed)

        self.ACNet = ACNetwork(obs, actions).to(device)
        self.optimizer = optim.Adam(self.ACNet.parameters(), lr=p.ALR)

    def get_state_dict(self):
        return self.ACNet.state_dict()

    def load_state_dict(self, f):
        self.ACNet.load_state_dict(torch.load(f))

    """ Decide what to do given a state """
    def select_action(self, obs, return_dist = True):
        # TODO: CHECK IF IT HAS 4-D (BATCH OR PARALLEL AGENTS) FOR NOW LET'S JUST ADD IT MANUALLY BECAUSE I KNOW I'M JUST SENDING ONE SET OF OBS
        # From np to tensor
        if obs.ndim == 4:
            obs = torch.FloatTensor(obs).to(device)                 # If it's already batcherized
        else:
            obs = torch.FloatTensor(obs).unsqueeze(0).to(device)    # Batcherize it so it has batch X channel X w X h
        probs, value = self.ACNet(obs)                   # Get probs from the Actor and the v(s) from the Critic
        dist = Categorical(probs)                               # create a categorical distribution given the probs
        action = dist.sample()                                  # sample an action
        logprob = dist.log_prob(action).unsqueeze(0)            # get the log p of the action
        entropy = dist.entropy().mean().unsqueeze(0)            # get H(π(a|s))
        #print(probs, value, action, logprob, entropy)
        if action.size(0) == 1:                                 # If we only have one action
            return action.item(), logprob, value, entropy
        else:                                                   # if we have mutliple actions either because we're using many workers or having parallel simulated trajectories
            return action, logprob, value, entropy

    """ Calculate the actual returns according to what was observed during the rollout """
    def analyze_rollout(self, state, memory): #rolloutRewards, dones):
        # Retrieve from memory
        rolloutRewards = memory.retrieve('reward')
        dones = memory.retrieve('done')
        # First bootstrap the return calculation
        if dones[-1]:
            R = torch.tensor([0.0]).to(device)      # R=0 if it was terminal
        else:
            _, _, R, _ = self.select_action(state)  # R=V(s_t) otherwise, to estimate the value of this state

        #print(R, rolloutRewards, dones)
        returns = []
        rolloutRewards.reverse()    # They'll be calculated from
        dones.reverse()
        for r, d in zip(rolloutRewards, dones):
            # Go through each step along the rollout (in reverse) and at each of them calculate the return
            # G = V(s) = R_1 + γR_2 + γ^2R_3 + ...
            R = r + p.GAMMA * R * (1 - d)
            returns.append(R)
        returns.reverse()
        #print(returns)
        return returns

    """  sets up the loss functions in order to do backprop """
    def learn(self, memory, returns): #logprobs, returns, values, entropies):
        logprobs = torch.cat(memory.retrieve('logprob'))          # logp of each action taken during the rollout
        returns = torch.cat(returns).detach()   # detach because doesn't depend on any weights unlike logπ(a|s;θ) & V(s;ω)
        values = torch.cat(memory.retrieve('value'))              # V(s) estimates of each state seen during the rollout
        entropies = torch.cat(memory.retrieve('entropy'))        # H(π(A|S=s)) of each state observed during the rollout
        advantage = returns - values            # A(s,a) = Q(s,a) - V(s) = r + γV(s') + V(s) = TD error
        actor_loss = -(logprobs * advantage.detach()).mean()    # log π(a|s;θ) A(s,a)
        critic_loss = advantage.pow(2).mean()                   # A = [y - V(s)]^2 (a regression problem)
        # L = L_a + β L_c - λ H
        loss = actor_loss + (p.CWEIGHT * critic_loss) - (p.ENTROPY_LOSS * p.EWEIGHT * entropies.mean())
        #print(logprobs, returns, values, entropies)
        #print(advantage)
        # Backprop
        self.optimizer.zero_grad()  # set gradients to 0 so they won't accumulate since it isn't an RNN
        loss.backward()  # compute dLoss/dx (x.grad) for every parameter x

        nn.utils.clip_grad_norm_(self.ACNet.parameters(), p.MAX_GRADNORM)

        self.optimizer.step()  # update every param x according to the computed x.grad
        return loss

    """ tests an agent on the environment and reports back its average performance 
    and average steps needed to complete it"""
    def test(self, test_episodes=10, visualize=False):
        #test_env = gym.make(p.ENVIRONMENT)
        test_env = MiniPacman('regular', 1000)
        testRewards, testSteps = [], []
        for episode in range(test_episodes):
            state = test_env.reset()
            if visualize:
                test_env.render()
            done = False
            episodeReward = 0
            t = 0
            while not done:
                action, _, _, _ = self.select_action(state)
                state, reward, done, _ = test_env.step(action)
                if visualize:
                    test_env.render()
                episodeReward += reward
                t+=1
            testRewards.append(episodeReward)
            testSteps.append(t)
        return np.mean(testRewards), np.mean(testSteps)


class A2CDistill(A2C):

    def __init__(self, obs, actions):
        #super(A2CDistill, self).__init(obs, action)
        #A2C.__init__(self, obs, actions)
        #torch.manual_seed(0)
        #self.obs = obs  # input
        #self.actions = actions  # output
        # self.seed = random.seed(seed)
        self.ACNet = ACNetwork(obs, actions).to(device)
        self.optimizer = optim.Adam(self.ACNet.parameters(), lr=1e-8)

    def load_state_dict(self, mode):
        self.ACNet.load_state_dict(torch.load("distill_" + mode))

    """ Decide what to do given a state 
    this one in additon returns the whole logprob distribution, not just of the action"""
    def select_action(self, obs, return_dist=True):
        # TODO: CHECK IF IT HAS 4-D (BATCH OR PARALLEL AGENTS) FOR NOW LET'S JUST ADD IT MANUALLY BECAUSE I KNOW I'M JUST SENDING ONE SET OF OBS
        # From np to tensor
        if obs.ndim == 4:
            obs = torch.FloatTensor(obs).to(device)  # If it's already batcherized
        else:
            obs = torch.FloatTensor(obs).unsqueeze(0).to(device)  # Batcherize it so it has batch X channel X w X h
        probs, value = self.ACNet(obs)  # Get probs from the Actor and the v(s) from the Critic
        dist = Categorical(probs)  # create a categorical distribution given the probs
        action = dist.sample()  # sample an action
        logprob = dist.log_prob(action).unsqueeze(0)  # get the log p of the action
        entropy = dist.entropy().mean().unsqueeze(0)  # get H(π(a|s))
        # print(probs, value, action, logprob, entropy)
        if action.size(0) == 1:  # If we only have one action
            return action.item(), logprob, value, entropy, torch.log(probs)
        else:  # if we have mutliple actions either because we're using many workers or having parallel simulated trajectories
            return action, logprob, value, entropy, torch.log(probs)

    def distillation(self, memory):
        probs = torch.cat(memory.retrieve('probs'))
        distill_logprobs = torch.cat(memory.retrieve('distill_logprob'))
        # todo: is there a better way to do this sum mean
        # todo 2: this is actually be cross entropy:  pi log pi_imaginationaugmented
        # todo 3: shouldn't this loss be added to a total loss?
        #print('distillation')
        # print((probs.detach() * distill_logprobs))#.sum(1).mean())
        # #exit()
        # print(probs)
        # print(distill_logprobs)
        # exit()
        distill_loss = p.DISLOSS_WEIGHT * (probs.detach() * distill_logprobs).sum(1).mean()

        # print('BEFORE ', self.ACNet.conv[0].weight)
        # print('BEFORE grad ', self.ACNet.conv[0].weight.grad)
        self.optimizer.zero_grad()
        distill_loss.backward()

        nn.utils.clip_grad_norm_(self.ACNet.parameters(), p.MAX_GRADNORM)

        self.optimizer.step()
        # print('AFTER ', self.ACNet.conv[0].weight)
        # print('AFTER grad ', self.ACNet.conv[0].weight.grad)
        return distill_loss
