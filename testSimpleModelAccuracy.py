from minipacman.minipacman import MiniPacman
import minipacman.minipacman_utils as mpu

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.env_model_boot import EnvModel as envboot#, EnvModelNonProb
from models.env_model_rpf import EnvModel as envrpf
from a2c import A2C
from params import Params as p

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set up env
mode = 'regular'
env = MiniPacman(mode, 1000)

# Init env model
input_space = env.observation_space.shape
action_space = env.action_space.n
env_model = envboot(input_space, mpu.num_pixels, mpu.num_rewards, p.NUM_HEADS)
env_model_rpf = envrpf(input_space, mpu.num_pixels, mpu.num_rewards, p.NUM_HEADS)
#env_model_a2c = EnvModelNonProb(input_space, mpu.num_pixels, mpu.num_rewards)
fname = './worldModels/env_model_boot_rew_a2c'
env_model.load_state_dict(fname)
fname = './worldModels/env_model_rpf_rew_a2c'
env_model_rpf.load_state_dict(fname)


# Load agent (pre-trained policy)
agent = A2C(input_space, action_space)
agent.load_state_dict("a2c_" + mode)


# Numpy array - state shape (3,15,19)
done = False
state = env.reset()
state_torch = torch.FloatTensor(state).unsqueeze(0)

total_err_head_rpf = []
total_err_head_boot = []
total_err_composite_boot = []
total_err_composite_rpf = []
total_acc_composite_boot = []
total_acc_composite_rpf = []
total_acc_vote_boot = []
total_acc_vote_rpf = []
total_acc_sampled_boot = []
total_acc_sampled_rpf = []

total_err_head_rpf_rew = []
total_err_head_boot_rew = []
total_err_composite_boot_rew = []
total_err_composite_rpf_rew = []
total_acc_composite_boot_rew = []
total_acc_composite_rpf_rew = []
total_acc_vote_boot_rew = []
total_acc_vote_rpf_rew = []
total_acc_sampled_boot_rew = []
total_acc_sampled_rpf_rew = []

acc_samp_boot_fruit = []
acc_samp_rpf_fruit = []
acc_samp_boot_ghost = []
acc_samp_rpf_ghost = []
acc_samp_boot_pacman = []
acc_samp_rpf_pacman = []
acc_samp_boot_eaten = []
acc_samp_rpf_eaten = []

acc_vote_boot_fruit = []
acc_vote_rpf_fruit = []
acc_vote_boot_ghost = []
acc_vote_rpf_ghost = []
acc_vote_boot_pacman = []
acc_vote_rpf_pacman = []
acc_vote_boot_eaten = []
acc_vote_rpf_eaten = []

acc_compo_boot_fruit = []
acc_compo_rpf_fruit = []
acc_compo_boot_ghost = []
acc_compo_rpf_ghost = []
acc_compo_boot_pacman = []
acc_compo_rpf_pacman = []
acc_compo_boot_eaten = []
acc_compo_rpf_eaten = []

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
    #action, _, _, _ = agent.select_action(state)a
    action = np.random.randint(action_space)

    # ==== Imagination

    # Form state-action input
    onehot_action = torch.zeros(1, action_space, *input_space[1:])
    onehot_action[0, action] = 1
    state_action = torch.cat([state_torch, onehot_action], 1).to(device)
    # Send them and retrieve what the env model heads think it'll happen
    imagined_obs, imagined_reward = env_model.imagine(state_action, list(range(p.NUM_HEADS)))
    imagined_obs_rpf, imagined_reward_rpf = env_model_rpf.imagine(state_action, list(range(p.NUM_HEADS)))
    # Stack them
    imagined_obs = torch.stack(imagined_obs)
    imagined_reward = torch.stack(imagined_reward)
    imagined_obs_rpf = torch.stack(imagined_obs_rpf)
    imagined_reward_rpf = torch.stack(imagined_reward_rpf)

    # Clone prediction for future error comparison
    pred_obs_boot, pred_rew_boot = torch.clone(imagined_obs), torch.clone(imagined_reward)
    pred_obs_rpf, pred_rew_rpf = torch.clone(imagined_obs_rpf), torch.clone(imagined_reward_rpf)
    # Get REWARDS (per head)
    # Dim: heads X outputs (1, we're predicting one reward) X type of rewards
    # Get the maximum in dim 2 (type of reward) but transform it into softmax and get the index
    imagined_reward = F.softmax(imagined_reward, dim=2).max(2)[1].cpu()
    imagined_reward = imagined_reward.squeeze(1).data.numpy()
    imagined_reward_rpf = F.softmax(imagined_reward_rpf, dim=2).max(2)[1].cpu()
    imagined_reward_rpf = imagined_reward_rpf.squeeze(1).data.numpy()
    # Get FRAMES (per head) (e.g. from 10,285,7 to 10,285)
    # Dim: heads X outputs (285, we're predicting 285 pixels) X type of pix
    # First clone im state to start to prepare it for graphical mode
    imagined_image = F.softmax(imagined_obs, dim=2).max(2)[1]
    imagined_image_rpf = F.softmax(imagined_obs_rpf, dim=2).max(2)[1]


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
    # 1) Get individual head errors then avg them
    err_head_rpf, err_head_boot = [], []
    for h in range(p.NUM_HEADS):    # Get err for each head
        err_head_boot.append(criterion(pred_obs_boot[h], target_state))
        err_head_rpf.append(criterion(pred_obs_rpf[h], target_state))
    total_err_head_rpf.append(torch.stack(err_head_rpf).mean().item())
    total_err_head_boot.append(torch.stack(err_head_boot).mean().item())
    # 2) Avg activations across heads first then get error from unification
    compo_obs_boot = torch.mean(pred_obs_boot, dim=0)
    compo_obs_rpf = torch.mean(pred_obs_rpf, dim=0)
    total_err_composite_boot.append(criterion(compo_obs_boot, target_state).item())
    total_err_composite_rpf.append(criterion(compo_obs_rpf, target_state).item())
    compo_obs_boot = F.softmax(compo_obs_boot, dim=1).max(1)[1]
    compo_obs_rpf = F.softmax(compo_obs_rpf, dim=1).max(1)[1]
    total_acc_composite_boot.append((compo_obs_boot == target_state).to(torch.float32).mean().item())
    total_acc_composite_rpf.append((compo_obs_rpf == target_state).to(torch.float32).mean().item())
    # 3) Majority vote
    vote_obs_boot = torch.mode(imagined_image, dim=0)[0]
    vote_obs_rpf = torch.mode(imagined_image_rpf, dim=0)[0]
    # Get accuracies
    total_acc_vote_boot.append((vote_obs_boot == target_state).to(torch.float32).mean().item())
    total_acc_vote_rpf.append((vote_obs_rpf == target_state).to(torch.float32).mean().item())
    # 4) Sampled
    sampled_obs_boot = torch.tensor(np.choose(np.random.randint(imagined_image.shape[0], size=imagined_image.shape[1]),
                                              imagined_image.cpu().numpy())).to(device)
    sampled_obs_rpf = torch.tensor(np.choose(np.random.randint(imagined_image_rpf.shape[0], size=imagined_image_rpf.shape[1]),
                  imagined_image_rpf.cpu().numpy())).to(device)
    total_acc_sampled_boot.append((sampled_obs_boot == target_state).to(torch.float32).mean().item())
    total_acc_sampled_rpf.append((sampled_obs_rpf == target_state).to(torch.float32).mean().item())

    # Get Class accuracy (Fruit, Pacman, Ghost, Eaten cell)
    # They'll have nans whenever the selected class is not present in current reality
    acc_samp_boot_fruit.append((sampled_obs_boot[(target_state == FRUIT).nonzero()]==FRUIT).to(torch.float32).mean().item())
    acc_samp_rpf_fruit.append((sampled_obs_rpf[(target_state == FRUIT).nonzero()]==FRUIT).to(torch.float32).mean().item())
    acc_samp_boot_ghost.append((sampled_obs_boot[(target_state == GHOST).nonzero()]==GHOST).to(torch.float32).mean().item())
    acc_samp_rpf_ghost.append((sampled_obs_rpf[(target_state == GHOST).nonzero()]==GHOST).to(torch.float32).mean().item())
    acc_samp_boot_pacman.append((sampled_obs_boot[(target_state == PACMAN).nonzero()]==PACMAN).to(torch.float32).mean().item())
    acc_samp_rpf_pacman.append((sampled_obs_rpf[(target_state == PACMAN).nonzero()]==PACMAN).to(torch.float32).mean().item())
    acc_samp_boot_eaten.append((sampled_obs_boot[(target_state == EATEN_CELL).nonzero()]==EATEN_CELL).to(torch.float32).mean().item())
    acc_samp_rpf_eaten.append((sampled_obs_rpf[(target_state == EATEN_CELL).nonzero()]==EATEN_CELL).to(torch.float32).mean().item())

    acc_compo_boot_fruit.append(
        (compo_obs_boot[(target_state == FRUIT).nonzero()] == FRUIT).to(torch.float32).mean().item())
    acc_compo_rpf_fruit.append(
        (compo_obs_rpf[(target_state == FRUIT).nonzero()] == FRUIT).to(torch.float32).mean().item())
    acc_compo_boot_ghost.append(
        (compo_obs_boot[(target_state == GHOST).nonzero()] == GHOST).to(torch.float32).mean().item())
    acc_compo_rpf_ghost.append(
        (compo_obs_rpf[(target_state == GHOST).nonzero()] == GHOST).to(torch.float32).mean().item())
    acc_compo_boot_pacman.append(
        (compo_obs_boot[(target_state == PACMAN).nonzero()] == PACMAN).to(torch.float32).mean().item())
    acc_compo_rpf_pacman.append(
        (compo_obs_rpf[(target_state == PACMAN).nonzero()] == PACMAN).to(torch.float32).mean().item())
    acc_compo_boot_eaten.append(
        (compo_obs_boot[(target_state == EATEN_CELL).nonzero()] == EATEN_CELL).to(torch.float32).mean().item())
    acc_compo_rpf_eaten.append(
        (compo_obs_rpf[(target_state == EATEN_CELL).nonzero()] == EATEN_CELL).to(torch.float32).mean().item())

    acc_vote_boot_fruit.append(
        (vote_obs_boot[(target_state == FRUIT).nonzero()] == FRUIT).to(torch.float32).mean().item())
    acc_vote_rpf_fruit.append(
        (vote_obs_rpf[(target_state == FRUIT).nonzero()] == FRUIT).to(torch.float32).mean().item())
    acc_vote_boot_ghost.append(
        (vote_obs_boot[(target_state == GHOST).nonzero()] == GHOST).to(torch.float32).mean().item())
    acc_vote_rpf_ghost.append(
        (vote_obs_rpf[(target_state == GHOST).nonzero()] == GHOST).to(torch.float32).mean().item())
    acc_vote_boot_pacman.append(
        (vote_obs_boot[(target_state == PACMAN).nonzero()] == PACMAN).to(torch.float32).mean().item())
    acc_vote_rpf_pacman.append(
        (vote_obs_rpf[(target_state == PACMAN).nonzero()] == PACMAN).to(torch.float32).mean().item())
    acc_vote_boot_eaten.append(
        (vote_obs_boot[(target_state == EATEN_CELL).nonzero()] == EATEN_CELL).to(torch.float32).mean().item())
    acc_vote_rpf_eaten.append(
        (vote_obs_rpf[(target_state == EATEN_CELL).nonzero()] == EATEN_CELL).to(torch.float32).mean().item())

    # print(target_state.view((15,19))) # 0 - fruit, 6 (ghost), 1 (pcman), 5 (eaten cell)

    # Reward
    # 2) Avg activations across heads first then get error from unification
    compo_rew_boot = torch.mean(pred_rew_boot, dim=0)
    compo_rew_rpf = torch.mean(pred_rew_rpf, dim=0)
    compo_rew_boot = F.softmax(compo_rew_boot, dim=1).max(1)[1]
    compo_rew_rpf = F.softmax(compo_rew_rpf, dim=1).max(1)[1]
    total_acc_composite_boot_rew.append((compo_rew_boot == reward).to(torch.float32).mean().item())
    total_acc_composite_rpf_rew.append((compo_rew_rpf == reward).to(torch.float32).mean().item())
    # 3) Majority vote
    from scipy import stats
    vote_rew_boot = stats.mode(imagined_reward)[0]
    vote_rew_rpf = stats.mode(imagined_reward_rpf)[0]
    total_acc_vote_boot_rew.append((vote_rew_boot == reward).mean())
    total_acc_vote_rpf_rew.append((vote_rew_rpf == reward).mean())
    # 4) Sampled
    sampled_rew_boot = np.random.choice(imagined_reward, 1)
    sampled_rew_rpf = np.random.choice(imagined_reward_rpf, 1)
    total_acc_sampled_boot_rew.append((sampled_rew_boot == reward).mean().item())
    total_acc_sampled_rpf_rew.append((sampled_rew_rpf == reward).mean().item())

    if done:
        print(rewardsEpi)
        state = env.reset()
        state_torch = torch.FloatTensor(state).unsqueeze(0)
        episode +=1
        rewardsEpi = 0
        # Results so far
        print('Frame prediction')
        print('Episode {} Head avg error Boot: {} RPF: {}'.format(episode, np.mean(total_err_head_boot), np.mean(total_err_head_rpf)))
        print('Episode {} Unified activation error Boot: {} RPF: {}'.format(episode, np.mean(total_err_composite_boot), np.mean(total_err_composite_rpf)))
        print('Episode {} Unified activation accuracy Boot: {} RPF: {}'.format(episode, np.mean(total_acc_composite_boot), np.mean(total_acc_composite_rpf)))
        print('Episode {} Majority vote accuracy Boot: {} RPF: {}'.format(episode, np.mean(total_acc_vote_boot), np.mean(total_acc_vote_rpf)))
        print('Episode {} Sampling accuracy Boot: {} RPF: {}'.format(episode, np.mean(total_acc_sampled_boot), np.mean(total_acc_sampled_rpf)))
        print('Reward prediction')
        print('Episode {} Unified activation accuracy Boot: {} RPF: {}'.format(episode, np.mean(total_acc_composite_boot_rew), np.mean(total_acc_composite_rpf_rew)))
        print('Episode {} Majority vote accuracy Boot: {} RPF: {}'.format(episode, np.mean(total_acc_vote_boot_rew), np.mean(total_acc_vote_rpf_rew)))
        print('Episode {} Sampling accuracy Boot: {} RPF: {}'.format(episode, np.mean(total_acc_sampled_boot_rew), np.mean(total_acc_sampled_rpf_rew)))

        print('Fruit prediction')
        print('Unified activation Boot: {} RPF: {}'.format(
            np.mean(np.array(acc_compo_boot_fruit)[~np.isnan(np.array(acc_compo_boot_fruit))]),
            np.mean(np.array(acc_compo_rpf_fruit)[~np.isnan(np.array(acc_compo_rpf_fruit))])
        ))
        print('Majority vote Boot: {} RPF: {}'.format(
            np.mean(np.array(acc_vote_boot_fruit)[~np.isnan(np.array(acc_vote_boot_fruit))]),
            np.mean(np.array(acc_vote_rpf_fruit)[~np.isnan(np.array(acc_vote_rpf_fruit))])
        ))
        print('Sampling Boot: {} RPF: {}'.format(
            np.mean(np.array(acc_samp_boot_fruit)[~np.isnan(np.array(acc_samp_boot_fruit))]),
            np.mean(np.array(acc_samp_rpf_fruit)[~np.isnan(np.array(acc_samp_rpf_fruit))])
        ))

        print('Ghost prediction')
        print('Unified activation Boot: {} RPF: {}'.format(
            np.mean(np.array(acc_compo_boot_ghost)[~np.isnan(np.array(acc_compo_boot_ghost))]),
            np.mean(np.array(acc_compo_rpf_ghost)[~np.isnan(np.array(acc_compo_rpf_ghost))])
        ))
        print('Majority vote Boot: {} RPF: {}'.format(
            np.mean(np.array(acc_vote_boot_ghost)[~np.isnan(np.array(acc_vote_boot_ghost))]),
            np.mean(np.array(acc_vote_rpf_ghost)[~np.isnan(np.array(acc_vote_rpf_ghost))])
        ))
        print('Sampling Boot: {} RPF: {}'.format(
            np.mean(np.array(acc_samp_boot_ghost)[~np.isnan(np.array(acc_samp_boot_ghost))]),
            np.mean(np.array(acc_samp_rpf_ghost)[~np.isnan(np.array(acc_samp_rpf_ghost))])
        ))

        print('Pacman prediction')
        print('Unified activation Boot: {} RPF: {}'.format(
            np.mean(np.array(acc_compo_boot_pacman)[~np.isnan(np.array(acc_compo_boot_pacman))]),
            np.mean(np.array(acc_compo_rpf_pacman)[~np.isnan(np.array(acc_compo_rpf_pacman))])
        ))
        print('Majority vote Boot: {} RPF: {}'.format(
            np.mean(np.array(acc_vote_boot_pacman)[~np.isnan(np.array(acc_vote_boot_pacman))]),
            np.mean(np.array(acc_vote_rpf_pacman)[~np.isnan(np.array(acc_vote_rpf_pacman))])
        ))
        print('Sampling Boot: {} RPF: {}'.format(
            np.mean(np.array(acc_samp_boot_pacman)[~np.isnan(np.array(acc_samp_boot_pacman))]),
            np.mean(np.array(acc_samp_rpf_pacman)[~np.isnan(np.array(acc_samp_rpf_pacman))])
        ))

        print('Eaten prediction')
        print('Unified activation Boot: {} RPF: {}'.format(
            np.mean(np.array(acc_compo_boot_eaten)[~np.isnan(np.array(acc_compo_boot_eaten))]),
            np.mean(np.array(acc_compo_rpf_eaten)[~np.isnan(np.array(acc_compo_rpf_eaten))])
        ))
        print('Majority vote Boot: {} RPF: {}'.format(
            np.mean(np.array(acc_vote_boot_eaten)[~np.isnan(np.array(acc_vote_boot_eaten))]),
            np.mean(np.array(acc_vote_rpf_eaten)[~np.isnan(np.array(acc_vote_rpf_eaten))])
        ))
        print('Sampling Boot: {} RPF: {}'.format(
            np.mean(np.array(acc_samp_boot_eaten)[~np.isnan(np.array(acc_samp_boot_eaten))]),
            np.mean(np.array(acc_samp_rpf_eaten)[~np.isnan(np.array(acc_samp_rpf_eaten))])
        ))
        print("===============================================================================")

