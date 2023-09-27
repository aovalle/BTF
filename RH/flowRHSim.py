'''this is with access to the simulator'''

import sys
sys.path.append('../')
from utils.plotting import plot_rew_time
from utils.openai.multiprocessing_env import SubprocVecEnv
from params import Params as p
from rolling_horizon import RollingHorizon

from Model.env_model import EnvModel
from minipacman.minipacman import MiniPacman
import minipacman.minipacman_utils as mpu
from utils.misc import one_hot_encoding
from tensorboardX import SummaryWriter
import re
import numpy as np
import copy

from os.path import join
from torchvision.utils import save_image

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TEST_EVERY = 100

# Init Env
num_envs = 1
#env = gym.make(p.ENVIRONMENT)
mode = "regular"
env = MiniPacman(mode, 1000)
simulator = MiniPacman(mode, 1000)
#env.seed(0)

input_space = env.observation_space.shape
action_space = env.action_space.n

print(input_space, action_space, mpu.num_pixels, mpu.num_rewards)

# Init env model
env_model = EnvModel(input_space, mpu.num_pixels, mpu.num_rewards)
env_model.load_state_dict(mode, kind="-a2c_icm")

# Init RH
agentRH = RollingHorizon(input_space, action_space, env_model)

t = 0
state = env.reset()

state_image = torch.FloatTensor(state).unsqueeze(0) #.permute(1, 2, 0).cpu().numpy()
save_image(state_image, join('rh-samples/resetstate.png'))

# for plotting
t_plot, r_plot, step_plot = [], [], []
epRewards = 0
rewards = []
while t < p.RH_EVAL_STEPS:

    t += 1
    oldState = state  # stores s_t since we're about to get s_t+1

    # Get a copy of the world state
    cloned_state = copy.deepcopy(env.env.world_state)
    simulator.env.world_state = cloned_state    # Set simulator to that state
    ws = env.env.world_state
    ws2 = simulator.env.world_state
    #print(np.array_equal(ws['food'], ws2['food']))
    # print(ws)
    # print(ws2)

    comp_state = copy.deepcopy(env.env.world_state)
    action, rollout = agentRH.select_action(state, simulator, cloned_state, comp_state)
    #action = np.random.randint(action_space)

    state, reward, done, _ = env.step(action)
    epRewards += reward

    #state2, reward2, done2, _ = simulator.step(action)
    ws = env.env.world_state
    ws2 = simulator.env.world_state

    #print(reward == reward2)
    #print(ws)
    #print(ws2)
    #print(np.array_equal(ws['food'], ws2['food']))


    if done:
        state = env.reset()
        rewards.append(epRewards)

        print('episode rewards ', epRewards, ' time steps ', t)
        r_plot.append(epRewards)
        #exit()
        print("So far ", t, " ", np.mean(r_plot), np.std(r_plot))

        epRewards = 0
        if p.EPISODIC:
            break

    # if t % TEST_EVERY == 0 and len(rewards) > 0:
    #     meanRew = np.mean(rewards)
    #     rewards = []
    #     #print('Time step {}\tAverage Score: {:.2f}'.format(t, meanRew))
    #     testRewards, testSteps = agentRH.test(test_episodes=2)
    #     print('Time step {}\tAverage Score: {:.2f}\tAverage Steps: {:.2f}'.format(t, testRewards, testSteps))
    #     # tensorboard
    #     tbWriter.add_scalar('Reward per episode', testRewards, t)
    #     tbWriter.add_scalar('Steps per episode', testSteps, t)
    #     t_plot.append(t)
    #     r_plot.append(testRewards)
    #     step_plot.append(testSteps)

tbWriter.close()
plot_rew_time(t_plot, r_plot)
exit()