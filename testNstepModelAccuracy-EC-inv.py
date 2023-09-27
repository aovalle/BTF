'''
10 march 2020
This version tries to test accuracy on n-step predictions for
multi-headed architectures (both non-EC and EC)
'''

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
from error_correction import correct_missing, correct_additional
from utils.misc import prepAndSaveImg

from scipy import stats
import os
from os.path import join
from torchvision.utils import save_image

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
fname = './worldModels/env_model_boot_rand_25k'
env_model.load_state_dict(fname)
fname = './worldModels/env_model_rpf_rand_25k'
env_model_rpf.load_state_dict(fname)


# Load agent (pre-trained policy)
agent = A2C(input_space, action_space)
agent.load_state_dict("a2c_" + mode)


# Numpy array - state shape (3,15,19)
done = False
state = env.reset()
state_torch = torch.FloatTensor(state).unsqueeze(0)
old_state_cat = np.expand_dims(state, axis=0) if state.ndim != 4 else state
old_state_cat = torch.LongTensor(mpu.pix_to_target(old_state_cat))#.to(device)

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
#UNEATEN_CELL = ?

g_curr_pos = env.get_ghost_pos()
g_curr_pos = 15*g_curr_pos[0] + g_curr_pos[1]
p_curr_pos = env.get_pac_pos()
p_curr_pos = 15*p_curr_pos[0] + p_curr_pos[1]

# LOOP
EPISODES = 100
episode = 0
rewardsEpi = 0

POSTPROCESS = False
SEQUENCE = 1
#POL = 'a2c'
POL = 'rand'
RPF = True
BT = 2
BOOT_TYPE = ['compo', 'vote', 'sample']

rstates, istates, actions, rrew, irew = [], [], [], [], []
countFrame = {FRUIT:0, GHOST:0, PACMAN:0}
nframes = 0
while episode < EPISODES:

    obs = np.copy(state)
    obs_torch = torch.FloatTensor(obs).unsqueeze(0)

    #note
    try:
        g_curr_pos = env.get_ghost_pos()
        g_curr_pos = 15 * g_curr_pos[0] + g_curr_pos[1]
    except:
        g_curr_pos = g_curr_pos
    try:
        p_curr_pos = env.get_pac_pos()
        p_curr_pos = 15 * p_curr_pos[0] + p_curr_pos[1]
    except:
        p_curr_pos = p_curr_pos

    old_state_cat = np.expand_dims(state, axis=0) if state.ndim != 4 else state
    old_state_cat = torch.LongTensor(mpu.pix_to_target(old_state_cat)).to(device)


    for step in range(SEQUENCE):
        # Select action according to policy
        if POL == 'a2c':
            action, _, _, _ = agent.select_action(obs)
        elif POL == 'rand':
            action = np.random.randint(action_space)

        # ==== Real World
        state, reward, done, _ = env.step(action)
        state_torch = torch.FloatTensor(state).unsqueeze(0)
        rewardsEpi += reward

        # ==== Store real state and reward (class)
        # convert to class form
        real_state = np.expand_dims(state, axis=0) if state.ndim != 4 else state
        real_state = np.array(mpu.pix_to_target(real_state))
        # print(target_state == imagined_image.cpu().numpy())
        rstates.append(real_state)
        #rrew.append(mpu.rewards_to_target(mode, reward)[0])
        rrew.append(reward)
        actions.append(action)

        if done:
            state = env.reset()
            state_torch = torch.FloatTensor(state).unsqueeze(0)
            episode += 1
            rewardsEpi = 0

            # Results so far
            print('Episode {}'.format(episode))
            print('Frame accuracy: {}'.format(np.mean(total_acc_obs)))
            print('Reward accuracy: {}'.format(np.mean(total_acc_rew)))
            print('Pacman accuracy: {}'.format(np.mean(np.array(acc_pacman)[~np.isnan(np.array(acc_pacman))])))
            print('Ghost accuracy: {}'.format(np.mean(np.array(acc_ghost)[~np.isnan(np.array(acc_ghost))])))
            print('Fruit accuracy: {}'.format(np.mean(np.array(acc_fruit)[~np.isnan(np.array(acc_fruit))])))
            print('Eaten cell accuracy: {}'.format(np.mean(np.array(acc_eaten)[~np.isnan(np.array(acc_eaten))])))

            break

    for ste in range(step+1):

        # ==== Imagination
        # Form state-action input
        onehot_action = torch.zeros(1, action_space, *input_space[1:])
        onehot_action[0, actions[ste]] = 1
        state_action = torch.cat([obs_torch, onehot_action], 1).to(device)
        # Send them and retrieve what the env model heads think it'll happen
        if not RPF:
            imagined_obs, imagined_reward = env_model.imagine(state_action, list(range(p.NUM_HEADS)))
            # Stack the predictions of each head so we have HEADS x 285 x PIX CLASSES (7)
            imagined_obs = torch.stack(imagined_obs)
            imagined_reward = torch.stack(imagined_reward)
            # Clone prediction for future error comparison
            pred_obs_boot, pred_rew_boot = torch.clone(imagined_obs), torch.clone(imagined_reward)
            # Get REWARDS (per head)
            # Dim: heads X outputs (1, we're predicting one reward) X type of rewards
            # Get the maximum in dim 2 (type of reward) but transform it into softmax and get the index
            imagined_reward = F.softmax(imagined_reward, dim=2).max(2)[1].cpu()
            imagined_reward = imagined_reward.squeeze(1).data.numpy()
            # Get FRAMES (per head) (e.g. from 10,285,7 to 10,285)
            # Dim: heads X outputs (285, we're predicting 285 pixels) X type of pix
            # First clone im state to start to prepare it for graphical mode
            imagined_image = F.softmax(imagined_obs, dim=2).max(2)[1]

            # Store imagined state, imagined reward and actions
            if BOOT_TYPE[BT] == 'compo':
                compo_obs_boot = torch.mean(pred_obs_boot, dim=0)
                im_obs_boot = F.softmax(compo_obs_boot, dim=1).max(1)[1]
                im_rew_boot = torch.mean(pred_rew_boot, dim=0)
                im_rew_boot = mpu.mode_rewards['regular'][F.softmax(im_rew_boot, dim=1).max(1)[1]]
            elif BOOT_TYPE[BT] == 'vote':
                im_obs_boot = torch.mode(imagined_image, dim=0)[0]
                im_rew_boot = mpu.mode_rewards['regular'][stats.mode(imagined_reward)[0].item()]
            elif BOOT_TYPE[BT] == 'sample':
                im_obs_boot = torch.tensor(np.choose(np.random.randint(imagined_image.shape[0], size=imagined_image.shape[1]),imagined_image.cpu().numpy())).to(device)
                im_rew_boot = mpu.mode_rewards['regular'][np.random.choice(imagined_reward, 1).item()]

            if POSTPROCESS:
                # If there was no ghost
                if (im_obs_boot == p.GHOST).sum().item() == 0:
                    # First check that if there is no ghost predicted, maybe it's because
                    # the ghost is dead (or yellow due to fruit) in that case don't EC
                    # So check against last known real obs
                    if (old_state_cat == p.GHOST).sum().item() > 0:
                        # Get an EC frame
                        #im_obs_boot, g_pos = correct_missing(im_obs_boot, imagined_image, p.GHOST, pred, g_curr_pos, True)
                        im_obs_boot, g_curr_pos = correct_missing(im_obs_boot, imagined_image, p.GHOST, BOOT_TYPE[BT], g_curr_pos, True)
                # TODO: SHOULD THIS BE AN ELIF?
                #  IF NO THEN G_CURR_POS ABOVE SHOULD BE g_pos
                # If there was more than one ghost
                elif (im_obs_boot == p.GHOST).sum().item() > 1:
                    #im_obs_boot, g_pos = correct_additional(im_obs_boot, imagined_image, p.GHOST, pred, old_state_cat, True)
                    im_obs_boot, g_curr_pos = correct_additional(im_obs_boot, imagined_image, p.GHOST, BOOT_TYPE[BT], old_state_cat, True)

                if (im_obs_boot == p.PACMAN).sum().item() == 0:
                    #im_obs_boot, p_pos = correct_missing(im_obs_boot, imagined_image, p.PACMAN, pred, p_curr_pos, True)
                    im_obs_boot, p_curr_pos = correct_missing(im_obs_boot, imagined_image, p.PACMAN, BOOT_TYPE[BT], p_curr_pos, True)
                elif (im_obs_boot == p.PACMAN).sum().item() > 1:
                    #im_obs_boot, p_pos = correct_additional(im_obs_boot, imagined_image, p.PACMAN, pred, old_state_cat, True)
                    im_obs_boot, p__curr_pos = correct_additional(im_obs_boot, imagined_image, p.PACMAN, BOOT_TYPE[BT], old_state_cat, True)
                    # try:
                    #     im_obs_boot, p__curr_pos = correct_additional(im_obs_boot, imagined_image, p.PACMAN, BOOT_TYPE[BT], old_state_cat, True)
                    # except:
                    #     print(im_obs_boot)
                    #     print(imagined_image)
                    #     print(old_state_cat)
                    #     im_obs_boot, p__curr_pos = correct_additional(im_obs_boot, imagined_image, p.PACMAN, BOOT_TYPE[BT], old_state_cat, True)
                    #     exit()

                # TODO: WHAT SHOULD BE THE INITIALIZATION OF g_pos/p_pos
                #g_curr_pos = g_pos
                #p_curr_pos = p_pos

            old_state_cat = im_obs_boot.clone()
            istates.append(im_obs_boot.cpu().numpy())
            irew.append(im_rew_boot)
            obs = mpu.target_to_pix(im_obs_boot.cpu().numpy())
        elif RPF:
            imagined_obs_rpf, imagined_reward_rpf = env_model_rpf.imagine(state_action, list(range(p.NUM_HEADS)))
            imagined_obs_rpf = torch.stack(imagined_obs_rpf)
            imagined_reward_rpf = torch.stack(imagined_reward_rpf)
            pred_obs_rpf, pred_rew_rpf = torch.clone(imagined_obs_rpf), torch.clone(imagined_reward_rpf)
            imagined_reward_rpf = F.softmax(imagined_reward_rpf, dim=2).max(2)[1].cpu()
            imagined_reward_rpf = imagined_reward_rpf.squeeze(1).data.numpy()
            imagined_image_rpf = F.softmax(imagined_obs_rpf, dim=2).max(2)[1]

            # TODO: GENERATE RPF REWARDS
            # Store imagined state, imagined reward and actions
            if BOOT_TYPE[BT] == 'compo':
                compo_obs_rpf = torch.mean(pred_obs_rpf, dim=0)
                im_obs_rpf = F.softmax(compo_obs_rpf, dim=1).max(1)[1]
                im_rew_rpf = torch.mean(pred_rew_rpf, dim=0)
                im_rew_rpf = mpu.mode_rewards['regular'][F.softmax(im_rew_rpf, dim=1).max(1)[1]]
            elif BOOT_TYPE[BT] == 'vote':
                im_obs_rpf = torch.mode(imagined_image_rpf, dim=0)[0]
                im_rew_rpf = mpu.mode_rewards['regular'][stats.mode(imagined_reward_rpf)[0].item()]
            elif BOOT_TYPE[BT] == 'sample':
                im_obs_rpf = torch.tensor(np.choose(np.random.randint(imagined_image_rpf.shape[0], size=imagined_image_rpf.shape[1]),
                              imagined_image_rpf.cpu().numpy())).to(device)
                im_rew_rpf = mpu.mode_rewards['regular'][np.random.choice(imagined_reward_rpf, 1).item()]

            if POSTPROCESS:
                # If there was no ghost
                if (im_obs_rpf == p.GHOST).sum().item() == 0:
                    # First check that if there is no ghost predicted, maybe it's because
                    # the ghost is dead (or yellow due to fruit) in that case don't EC
                    # So check against last known real obs
                    if (old_state_cat == p.GHOST).sum().item() > 0:
                        # Get an EC frame
                        #im_obs_boot, g_pos = correct_missing(im_obs_boot, imagined_image, p.GHOST, pred, g_curr_pos, True)
                        im_obs_rpf, g_curr_pos = correct_missing(im_obs_rpf, imagined_image_rpf, p.GHOST, BOOT_TYPE[BT], g_curr_pos, True)
                # TODO: SHOULD THIS BE AN ELIF?
                #  IF NO THEN G_CURR_POS ABOVE SHOULD BE g_pos
                # If there was more than one ghost
                elif (im_obs_rpf == p.GHOST).sum().item() > 1:
                    #im_obs_boot, g_pos = correct_additional(im_obs_boot, imagined_image, p.GHOST, pred, old_state_cat, True)
                    im_obs_rpf, g_curr_pos = correct_additional(im_obs_rpf, imagined_image_rpf, p.GHOST, BOOT_TYPE[BT], old_state_cat, True)

                if (im_obs_rpf == p.PACMAN).sum().item() == 0:
                    #im_obs_boot, p_pos = correct_missing(im_obs_boot, imagined_image, p.PACMAN, pred, p_curr_pos, True)
                    im_obs_rpf, p_curr_pos = correct_missing(im_obs_rpf, imagined_image_rpf, p.PACMAN, BOOT_TYPE[BT], p_curr_pos, True)
                elif (im_obs_rpf == p.PACMAN).sum().item() > 1:
                    #im_obs_boot, p_pos = correct_additional(im_obs_boot, imagined_image, p.PACMAN, pred, old_state_cat, True)
                    im_obs_rpf, p__curr_pos = correct_additional(im_obs_rpf, imagined_image_rpf, p.PACMAN, BOOT_TYPE[BT], old_state_cat, True)

                # TODO: WHAT SHOULD BE THE INITIALIZATION OF g_pos/p_pos
                #g_curr_pos = g_pos
                #p_curr_pos = p_pos

            old_state_cat = im_obs_rpf.clone()
            istates.append(im_obs_rpf.cpu().numpy())
            irew.append(im_rew_rpf)
            obs = mpu.target_to_pix(im_obs_rpf.cpu().numpy())


        # The imagined obs will be used to generate next imaginary ones
        # 3 15 19
        obs = torch.FloatTensor(obs.reshape(15, 19, 3)).unsqueeze(0)
        obs = obs.permute(0,3,1,2)
        obs = obs.squeeze(0).numpy()

        # # print(obs)
        # # print(obs.shape)
        # obs = torch.FloatTensor(mpu.pix_to_target(obs.cpu().numpy())).to(device)
        # print(obs)
        # print(obs.shape)
        # print(obs == im_obs_boot)
        # # 3 15 19
        # exit()


    # Compute accuracies
    nistates = np.array(istates)
    nrstates = np.array(rstates)
    total_acc_obs.append((np.array(istates) == np.array(rstates)).mean())
    nirew = np.array(irew)
    nrrew = np.array(rrew)
    total_acc_rew.append((nirew == nrrew).mean())

    acc_pacman.append((nistates[(nrstates == PACMAN).nonzero()] == PACMAN).mean())
    acc_ghost.append((nistates[(nrstates == GHOST).nonzero()] == GHOST).mean())
    acc_fruit.append((nistates[(nrstates == FRUIT).nonzero()] == FRUIT).mean())
    acc_eaten.append((nistates[(nrstates == EATEN_CELL).nonzero()] == EATEN_CELL).mean())

    # Want to know the number of frames constraints were violated
    for elem in [GHOST, PACMAN, FRUIT]:
        # Number of times it appears per frame
        constr = np.bincount((nrstates == elem).nonzero()[0])
        consti = np.bincount((nistates == elem).nonzero()[0])
        idxc = min(len(constr), len(consti))
        # Count the number of frames where constraints are being fulfilled (but that doesn't mean they're correct)
        # (e.g. Real frame has 1 magic pill, am i predicting exactly 1 magic pill)
        # corrconst = len(np.flatnonzero(constr[:idxc] == consti[:idxc]))
        countFrame[elem] += len(np.flatnonzero(constr[:idxc] == consti[:idxc]))

    # Total number of frames
    nframes += (step + 1)

    if done:
        print(np.array([*countFrame.values()]) / nframes, countFrame.values(), countFrame, nframes,
              ' FRUIT | GHOST | PACMAN')
        print("============================")

    rstates, istates, actions, rrew, irew = [], [], [], [], []
