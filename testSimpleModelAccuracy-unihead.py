from minipacman.minipacman import MiniPacman
import minipacman.minipacman_utils as mpu

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.env_model import EnvModel
from a2c import A2C
from params import Params as p

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set up env
mode = 'regular'
env = MiniPacman(mode, 1000)

# Init env model
input_space = env.observation_space.shape
action_space = env.action_space.n
env_model = EnvModel(input_space, mpu.num_pixels, mpu.num_rewards)
#env_model_a2c = EnvModelNonProb(input_space, mpu.num_pixels, mpu.num_rewards)
fname = './worldModels/unihead-cluster-bak-6march/env_model_uni_a2c_50k'
env_model.load_state_dict(fname)


# Load agent (pre-trained policy)
agent = A2C(input_space, action_space)
agent.load_state_dict("a2c_" + mode)


# Numpy array - state shape (3,15,19)
done = False
state = env.reset()
state_torch = torch.FloatTensor(state).unsqueeze(0)

total_acc_obs = []
total_acc_rew = []

acc_fruit = []
acc_ghost = []
acc_pacman = []
acc_eaten = []

# 0 - fruit, 6 (ghost), 1 (pcman), 5 (eaten cell)
FRUIT = 0
GHOST = 6
PACMAN = 1
EATEN_CELL = 5

# LOOP
EPISODES = 100
episode = 0
rewardsEpi = 0
while episode < EPISODES:

    # TODO: GET ENTROPY

    # Select action according to policy
    #action, _, _, _ = agent.select_action(state)
    action = np.random.randint(action_space)

    # ==== Imagination

    # Form state-action input
    onehot_action = torch.zeros(1, action_space, *input_space[1:])
    onehot_action[0, action] = 1
    state_action = torch.cat([state_torch, onehot_action], 1).to(device)
    # Send them and retrieve what the env model heads think it'll happen
    imagined_obs, imagined_reward = env_model.imagine(state_action)

    # Clone prediction for future error comparison
    pred_obs_boot, pred_rew_boot = torch.clone(imagined_obs), torch.clone(imagined_reward)

    # Get REWARDS (per head)
    # Dim: heads X outputs (1, we're predicting one reward) X type of rewards
    # Get the maximum in dim 2 (type of reward) but transform it into softmax and get the index
    imagined_reward = F.softmax(imagined_reward, dim=1).max(1)[1].cpu()
    imagined_reward = imagined_reward.data.numpy()
    # Get FRAMES (per head) (e.g. from 10,285,7 to 10,285)
    # Dim: heads X outputs (285, we're predicting 285 pixels) X type of pix
    # First clone im state to start to prepare it for graphical mode
    imagined_image = F.softmax(imagined_obs, dim=1).max(1)[1]


    # ==== Real World
    state, reward, done, _ = env.step(action)
    state_torch = torch.FloatTensor(state).unsqueeze(0)
    rewardsEpi += reward

    # ==== Prediction Error
    #Frame
    # Format and init
    target_state = np.expand_dims(state, axis=0) if state.ndim != 4 else state
    criterion = nn.CrossEntropyLoss()
    target_state = torch.LongTensor(mpu.pix_to_target(target_state)).to(device)

    # 3) Majority vote
    # Get accuracies
    total_acc_obs.append((imagined_image == target_state).to(torch.float32).mean().item())

    # Get Class accuracy (Fruit, Pacman, Ghost, Eaten cell)
    # They'll have nans whenever the selected class is not present in current reality
    acc_fruit.append((imagined_image[(target_state == FRUIT).nonzero()]==FRUIT).to(torch.float32).mean().item())
    acc_ghost.append((imagined_image[(target_state == GHOST).nonzero()]==GHOST).to(torch.float32).mean().item())
    acc_pacman.append((imagined_image[(target_state == PACMAN).nonzero()]==PACMAN).to(torch.float32).mean().item())
    acc_eaten.append((imagined_image[(target_state == EATEN_CELL).nonzero()]==EATEN_CELL).to(torch.float32).mean().item())

    # print(target_state.view((15,19))) # 0 - fruit, 6 (ghost), 1 (pcman), 5 (eaten cell)

    # Reward
    #target_reward = torch.LongTensor(mpu.rewards_to_target(mode, reward))#.to(device)
    target_reward = mpu.rewards_to_target(mode, reward)
    #total_acc_rew.append((imagined_reward == reward).mean())
    total_acc_rew.append((imagined_reward == target_reward).mean())

    if done:
        print(rewardsEpi)
        state = env.reset()
        state_torch = torch.FloatTensor(state).unsqueeze(0)
        episode +=1
        rewardsEpi = 0
        # Results so far
        print('Frame prediction')
        print('Episode {} accuracy: {}'.format(episode, np.mean(total_acc_obs)))
        print('Reward prediction')
        print('Episode {} accuracy: {}'.format(episode, np.mean(total_acc_rew)))

        print('Fruit prediction')
        print('Unified activation Boot: {}'.format(
            np.mean(np.array(acc_fruit)[~np.isnan(np.array(acc_fruit))])))

        print('Ghost prediction')
        print('Unified activation Boot: {}'.format(
            np.mean(np.array(acc_ghost)[~np.isnan(np.array(acc_ghost))])
        ))

        print('Pacman prediction')
        print('Unified activation Boot: {}'.format(
            np.mean(np.array(acc_pacman)[~np.isnan(np.array(acc_pacman))])
        ))

        print('Eaten prediction')
        print('Unified activation Boot: {}'.format(
            np.mean(np.array(acc_eaten)[~np.isnan(np.array(acc_eaten))])
        ))
        print("===============================================================================")

