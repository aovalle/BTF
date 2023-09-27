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

import os
from os.path import join
from torchvision.utils import save_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def saveImg(image, dir, fname):
    if type(image) is np.ndarray:
        img = np.copy(image)
        state_image = torch.FloatTensor(img).unsqueeze(0)  # .permute(1, 2, 0).cpu().numpy()
    elif type(image) is torch.Tensor:
        state_image = image.clone()

    if not os.path.isdir(dir):
        os.makedirs(dir)
    save_image(state_image, join(dir + fname))
def prepAndSaveImg(img, dir, fname):
    image = mpu.target_to_pix(img.data.cpu().numpy())
    image = torch.FloatTensor(image.reshape(15, 19, 3)).unsqueeze(0)
    # If i save it in something else that is not torch I SHOULD NOT DO THIS
    image = image.permute(0, 3, 1, 2)
    saveImg(image, dir, fname)

# Set up env
mode = 'regular'
env = MiniPacman(mode, 1000)

# Init env model
input_space = env.observation_space.shape
action_space = env.action_space.n
env_model = envboot(input_space, mpu.num_pixels, mpu.num_rewards, p.NUM_HEADS)
env_model_rpf = envrpf(input_space, mpu.num_pixels, mpu.num_rewards, p.NUM_HEADS)
#env_model_a2c = EnvModelNonProb(input_space, mpu.num_pixels, mpu.num_rewards)
fname = './worldModels/env_model_boot_rand_50k'
env_model.load_state_dict(fname)
fname = './worldModels/env_model_rpf_rand_50k'
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
#UNEATEN_CELL = ?

POSTPROCESS = True

# Ghost counters
g_compo_boot = []
g_compo_rpf = []
g_vote_boot = []
g_vote_rpf = []
g_samp_boot = []
g_samp_rpf = []
p_compo_boot = []
p_compo_rpf = []
p_vote_boot = []
p_vote_rpf = []
p_samp_boot = []
p_samp_rpf = []

acc_samp_boot_fruit_ec = []
acc_samp_rpf_fruit_ec = []
acc_samp_boot_ghost_ec = []
acc_samp_rpf_ghost_ec = []
acc_samp_boot_pacman_ec = []
acc_samp_rpf_pacman_ec = []
acc_samp_boot_eaten_ec = []
acc_samp_rpf_eaten_ec = []

acc_vote_boot_fruit_ec = []
acc_vote_rpf_fruit_ec = []
acc_vote_boot_ghost_ec = []
acc_vote_rpf_ghost_ec = []
acc_vote_boot_pacman_ec = []
acc_vote_rpf_pacman_ec = []
acc_vote_boot_eaten_ec = []
acc_vote_rpf_eaten_ec = []

acc_compo_boot_fruit_ec = []
acc_compo_rpf_fruit_ec = []
acc_compo_boot_ghost_ec = []
acc_compo_rpf_ghost_ec = []
acc_compo_boot_pacman_ec = []
acc_compo_rpf_pacman_ec = []
acc_compo_boot_eaten_ec = []
acc_compo_rpf_eaten_ec = []

g_curr_pos = env.get_ghost_pos()
g_curr_pos = 15*g_curr_pos[0] + g_curr_pos[1]
p_curr_pos = env.get_pac_pos()
p_curr_pos = 15*p_curr_pos[0] + p_curr_pos[1]




# LOOP
EPISODES = 100
episode = 0
rewardsEpi = 0
while episode < EPISODES:

    # TODO: GET ENTROPY

    # Select action according to policy
    action, _, _, _ = agent.select_action(state)
    #action = np.random.randint(action_space)

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
    try:
        g_next_pos = env.get_ghost_pos()
        g_next_pos = 15 * g_next_pos[0] + g_next_pos[1]
    except:
        g_next_pos = g_next_pos
    try:
        p_next_pos = env.get_pac_pos()
        p_next_pos = 15 * p_next_pos[0] + p_next_pos[1]
    except:
        p_next_pos = p_next_pos
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




    if POSTPROCESS:
        preds = ['sampled_obs_boot', 'vote_obs_boot', 'compo_obs_boot',
                 'sampled_obs_rpf', 'vote_obs_rpf', 'compo_obs_rpf']
        for pred in preds:
            # If there were no ghosts
            if (globals()[pred] == GHOST).sum().item() == 0:
                # Get positions where ghost is located in those where it's present
                if 'rpf' in pred:
                    positions = (imagined_image_rpf == GHOST).nonzero()
                else:
                    positions = (imagined_image == GHOST).nonzero()
                if 'sampled' in pred:
                    # Sample a position
                    try:
                        gpos = np.random.choice(positions[:, 1].cpu())
                    except:
                        gpos = g_curr_pos
                elif 'vote' or 'compo' in pred:
                    try:
                        gpos = torch.mode(positions[:,1])[0].item()
                    except:
                        gpos = g_curr_pos
                # Place the ghost
                globals()[pred][gpos] = GHOST
            # If there was more than one ghost
            elif (globals()[pred] == GHOST).sum().item() > 1:
                # Get where duplicated ghosts are located
                pred_positions = (globals()[pred] == GHOST).nonzero()
                print('Ghosts are in ', pred_positions)
                # Get predicted ghosts locations from heads
                if 'rpf' in pred:
                    positions = (imagined_image_rpf == GHOST).nonzero()
                    print('For those heads that decided ghost, in which positions?', positions)
                    #print(imagined_image_rpf[:, pred_positions[0]])
                    print('Values each head has in those positions where the composite is ghost ', imagined_image_rpf[:, pred_positions])
                else:
                    positions = (imagined_image == GHOST).nonzero()
                    print(positions)
                    #print(imagined_image[:, pred_positions[0]])
                    print(imagined_image[:, pred_positions])

                idx = np.isin(positions[:,1].cpu(), pred_positions.cpu())
                if 'sampled' in pred:
                    # Sample a position
                    gpos = np.random.choice(positions[idx, 1].cpu())
                elif 'vote' or 'compo' in pred:
                    gpos = torch.mode(positions[idx,1])[0].item()
                # Fill non-chosen positions with values from previous frame
                # Leave the selected ghost in-place and remove the others
                # print(globals()[pred])
                print('Position chosen for the ghost ', gpos)
                print('Positions non-chosen for ghosts ', pred_positions[pred_positions != gpos])

                prepAndSaveImg(globals()[pred], '.', 'preecghost.png')

                globals()[pred][pred_positions[pred_positions!=gpos]] = old_state_cat[pred_positions[pred_positions!=gpos]]
                print('Values were changed from ghosts to ', globals()[pred][pred_positions[pred_positions!=gpos]])
                duplicated = (globals()[pred][pred_positions[pred_positions!=gpos]] == GHOST).nonzero()
                print('However, if we check again are there ghosts still remaining? ', duplicated)

                leftover_ghost_pos = pred_positions[pred_positions != gpos][duplicated]
                print('If there are uncorrected ghosts, where are they? ', leftover_ghost_pos)

                # If there are still ghosts to be corrected let's check what can we extract from the heads
                if leftover_ghost_pos.nelement() > 0:
                    # Loop in case there are multiple uncorrected ghosts
                    for po in leftover_ghost_pos:
                        # Get then vector with every head prediction for this pixel
                        #print(p)
                        #print(imagined_image[:, p])
                        # Now from this vector select the one that is the most common among those that arent the ghost

                        # These are the values that aren't ghost and from which the new value will be selected
                        #print(imagined_image[:, p][imagined_image[:, p] != GHOST])

                        # NOTE: in the case of sampling it might be the case that the non chosen position might be full of ghost predictions
                        # for every head in that case there'll be no alternative (non ghost) EC pixel (i could select instead of sampling the one
                        # where all heads align)
                        # new value
                        # Sample or mode from the values that aren't the ghost
                        #print(torch.mode(imagined_image[:, p][imagined_image[:, p] != GHOST])[0].item())
                        #print(np.random.choice(imagined_image[:, p][imagined_image[:, p] != GHOST].cpu()))
                        if 'rpf' in pred:
                            if 'sampled' in pred:
                                try:
                                    newValue = np.random.choice(imagined_image_rpf[:, po][imagined_image_rpf[:, po] != GHOST].cpu()).item()
                                except:
                                    # NOTE: vector is full of ghosts thus nothing to sample, leave it as it is
                                    newValue = GHOST
                            elif 'vote' or 'compo' in pred:
                                try:
                                    newValue = torch.mode(imagined_image_rpf[:, po][imagined_image_rpf[:, po] != GHOST])[0].item()
                                except:
                                    newValue = GHOST
                        else:
                            if 'sampled' in pred:
                                try:
                                    newValue = np.random.choice(imagined_image[:, po][imagined_image[:, po] != GHOST].cpu()).item()
                                except:
                                    # NOTE: vector is full of ghosts thus nothing to sample, leave it as it is
                                    newValue = GHOST
                            elif 'vote' or 'compo' in pred:
                                try:
                                    newValue = torch.mode(imagined_image[:, po][imagined_image[:, po] != GHOST])[0].item()
                                except:
                                    newValue = GHOST


                        prepAndSaveImg(globals()[pred], '.', 'preecghost2nd.png')
                        globals()[pred][po] = newValue
                        prepAndSaveImg(globals()[pred], '.', 'postecghost2nd.png')


                    if leftover_ghost_pos.nelement() > 1:
                        print('need two corrections')
                        print(imagined_image[:, leftover_ghost_pos])
                        print(imagined_image[:, leftover_ghost_pos][imagined_image[:, leftover_ghost_pos] != GHOST])
                        exit()



                    # # Get then vector with every head prediction for this pixel
                    # print(imagined_image[:, leftover_ghost_pos])
                    # # Now from this vector select the one that is the most common among those that arent the ghost
                    #
                    # # This are the values that aren't ghost and from which the new value will be selected
                    # print(imagined_image[:, leftover_ghost_pos][imagined_image[:, leftover_ghost_pos] != GHOST])
                    #
                    # #NOTE: in the case of sampling it might be the case that the non chosen position might be full of ghost predictions
                    # # for every head in that case there'll be no alternative (non ghost) EC pixel (i could select instead of sampling the one
                    # # where all heads align)
                    # #new value
                    # # Sample or mode from the values that aren't the ghost
                    # # TODO IF: RPF OR BOOT
                    # print(torch.mode(imagined_image[:, leftover_ghost_pos][imagined_image[:, leftover_ghost_pos]!=GHOST])[0].item())
                    # print(np.random.choice(imagined_image[:, leftover_ghost_pos][imagined_image[:, leftover_ghost_pos]!=GHOST].cpu()))


                    #globals()[pred][leftover_ghost_pos] = new value



                    #exit()

                # TODO:
                #  - check that in the non-chosen positions you didn't copy a ghost
                #  - if you did then for that non-chosen position invoke what else was secondly predicted to be there (that isn't a ghost)
                # print(globals()[pred])
                # print(old_state_cat)
                # print(pred_positions[pred_positions!=gpos])
                # print(old_state_cat.shape)

                prepAndSaveImg(old_state_cat, '.', 'previousg.png')
                prepAndSaveImg(globals()[pred], '.', 'postecghost.png')

                #exit()

            # EC for PACMAN
            if (globals()[pred] == PACMAN).sum().item() == 0:
                # Get positions where ghost is located in those where it's present
                if 'rpf' in pred:
                    positions = (imagined_image_rpf == PACMAN).nonzero()
                else:
                    positions = (imagined_image == PACMAN).nonzero()
                if 'sampled' in pred:
                    # Sample a position
                    try:
                        ppos = np.random.choice(positions[:, 1].cpu())
                    except:
                        ppos = p_curr_pos
                elif 'vote' or 'compo' in pred:
                    try:
                        ppos = torch.mode(positions[:,1])[0].item()
                    except:
                        ppos = p_curr_pos
                # Place the ghost
                globals()[pred][ppos] = PACMAN
            # Check if there's more than one pacman in the frame
            elif (globals()[pred] == PACMAN).sum().item() > 1:
                # Get where duplicated pacmans are located
                pred_positions = (globals()[pred] == PACMAN).nonzero()
                # Get predicted ghosts locations from heads
                if 'rpf' in pred:
                    positions = (imagined_image_rpf == PACMAN).nonzero()
                else:
                    positions = (imagined_image == PACMAN).nonzero()
                # Get pacman overlap between heads and composite predictions
                idx = np.isin(positions[:, 1].cpu(), pred_positions.cpu())
                if 'sampled' in pred:
                    # Sample a position
                    ppos = np.random.choice(positions[idx, 1].cpu())
                elif 'vote' or 'compo' in pred:
                    ppos = torch.mode(positions[idx, 1])[0].item()

                prepAndSaveImg(old_state_cat, '.', 'previousp.png')
                prepAndSaveImg(globals()[pred], '.', 'preecpac.png')

                # TODO: Is this the best strategy for pacmans? it was for ghost because ghost
                #  doesn't eat fruits or cells
                #  it seems that if past real fram was also green then I should default to black
                #  not to just previous pixel

                # todo: probably i should have a condition if previous real frame was pacman
                #  then sure set to black if it wasn't set to previous pixel

                # Fill non-chosen positions with values from previous frame
                # Leave the selected pacman in-place and remove the others
                #globals()[pred][pred_positions[pred_positions != ppos]] = old_state_cat[pred_positions[pred_positions != ppos]]
                print('FROM')
                print(globals()[pred][pred_positions[pred_positions != ppos]])
                print('WFT')
                print(globals()[pred][pred_positions[pred_positions != ppos]][0])
                print(globals()[pred][pred_positions[pred_positions != ppos]][0])
                #globals()[pred][pred_positions[pred_positions != ppos]] = EATEN_CELL
                # Change to EATEN CELL only if previous real was PACMAN
                print(old_state_cat[pred_positions[pred_positions != ppos]] == PACMAN)
                print(old_state_cat[pred_positions[pred_positions != ppos]] != PACMAN)
                print('to change')
                print(globals()[pred][pred_positions[pred_positions != ppos]][old_state_cat[pred_positions[pred_positions != ppos]] == PACMAN])
                print(globals()[pred][pred_positions[pred_positions != ppos]][old_state_cat[pred_positions[pred_positions != ppos]] != PACMAN])


                #globals()[pred][pred_positions[pred_positions != ppos]][0] = 999 #EATEN_CELL
                # globals()[pred][pred_positions[pred_positions != ppos]][old_state_cat[pred_positions[pred_positions != ppos]] == PACMAN] = EATEN_CELL
                # globals()[pred][pred_positions[pred_positions != ppos]][old_state_cat[pred_positions[pred_positions != ppos]] != PACMAN] \
                #    = old_state_cat[pred_positions[pred_positions != ppos]]


                # NUMPY MODE ------------------------------------
                # vec = globals()[pred].data.cpu().numpy()
                # nonchosen = pred_positions[pred_positions != ppos].cpu().numpy()
                # #ten = vec[pred_positions[pred_positions != ppos]]
                # idx_pac = (old_state_cat[pred_positions[pred_positions != ppos]] == PACMAN).cpu().numpy()
                # idx_nopac = (old_state_cat[pred_positions[pred_positions != ppos]] != PACMAN).cpu().numpy()
                # pos_to_eaten = nonchosen[idx_pac]
                # pos_to_prev = nonchosen[idx_nopac]
                #
                # # print(vec[nonchosen], idx_pac, idx_nopac, pos_to_eaten, pos_to_prev, old_state_cat[pred_positions[pred_positions != ppos]])
                # # print(vec)
                # # print(np.atleast_1d(vec[nonchosen]), idx_pac, np.atleast_1d(vec[nonchosen])[idx_pac])
                #
                # #np.atleast_1d(vec[nonchosen])[0] = EATEN_CELL
                # vec[pos_to_eaten] = EATEN_CELL
                # vec[pos_to_prev] = old_state_cat[pred_positions[pred_positions != ppos]].cpu().numpy()
                # #np.atleast_1d(vec[nonchosen])[idx_nopac] = old_state_cat[pred_positions[pred_positions != ppos]].cpu().numpy()
                # # if vec[nonchosen].ndim == 0:
                # #     #if previous real frame was pacman then set to black
                # #     if old_state_cat[pred_positions[pred_positions != ppos]] == PACMAN:
                # #         vec[nonchosen] = EATEN_CELL
                # #     #otherwise just choose whatever was there
                # #     else:
                # #         vec[nonchosen] = old_state_cat[pred_positions[pred_positions != ppos]]
                # # else:
                # #     vec[nonchosen][idx_pac] = EATEN_CELL
                # #     vec[nonchosen][idx_nopac] = old_state_cat[pred_positions[pred_positions != ppos]]
                #
                # #print(ten, idx_pac, idx_nopac, old_state_cat[pred_positions[pred_positions != ppos]])
                # #ten[0] = EATEN_CELL
                # #ten[idx_pac] = EATEN_CELL
                # #ten[idx_nopac] = old_state_cat[pred_positions[pred_positions != ppos]]
                # #print(ten)
                #
                # vec = torch.tensor(vec)

                # TORCH MODE --------------------

                vec = globals()[pred]
                # Pacman positions not chosen to be kept
                nonchosen = pred_positions[pred_positions != ppos]
                # Indices of the
                idx_pac = (old_state_cat[pred_positions[pred_positions != ppos]] == PACMAN)
                idx_nopac = (old_state_cat[pred_positions[pred_positions != ppos]] != PACMAN)
                pos_to_eaten = nonchosen[idx_pac]
                pos_to_prev = nonchosen[idx_nopac]

                # print(vec[nonchosen], idx_pac, idx_nopac, pos_to_eaten, pos_to_prev, old_state_cat[pred_positions[pred_positions != ppos]])
                # print(vec)

                vec[pos_to_eaten] = EATEN_CELL
                vec[pos_to_prev] = old_state_cat[pred_positions[pred_positions != ppos]]

                # print(vec)

                prepAndSaveImg(vec, '.', 'postecpac.png')
                #prepAndSaveImg(globals()[pred], '.', 'postecpac.png')
                print('Exit at more than one pacman ', pred_positions)
                exit()

                # Now that it's been corrected we still have to check if there're pacmans left
                duplicated = (globals()[pred][pred_positions[pred_positions != ppos]] == PACMAN).nonzero()
                # Check where are those remaining pacmans
                leftover_pac_pos = pred_positions[pred_positions != ppos][duplicated]

                # TODO: IS THIS STILL NECESSARY?
                # If there are still pacmans to be corrected let's check what can we extract from the heads
                if leftover_pac_pos.nelement() > 0:
                    print('more than one pacman, so yes this is still necessary!!!!!')
                    exit()
                    # Loop in case there are multiple uncorrected pacmans
                    for po in leftover_pac_pos:

                        if 'rpf' in pred:
                            if 'sampled' in pred:
                                newValue = np.random.choice(imagined_image_rpf[:, po][imagined_image_rpf[:, po] != PACMAN].cpu())
                            elif 'vote' or 'compo' in pred:
                                newValue = torch.mode(imagined_image_rpf[:, po][imagined_image_rpf[:, po] != PACMAN])[0].item()
                        else:
                            if 'sampled' in pred:
                                newValue = np.random.choice(imagined_image[:, po][imagined_image[:, po] != PACMAN].cpu())
                            elif 'vote' or 'compo' in pred:
                                newValue = torch.mode(imagined_image[:, po][imagined_image[:, po] != PACMAN])[0].item()

                        prepAndSaveImg(globals()[pred], '.', 'preecpac2nd.png')
                        globals()[pred][po] = newValue
                        prepAndSaveImg(globals()[pred], '.', 'postecpac2nd.png')
                        print('Exit at leftover pacmans after correction')
                        exit()

                    if leftover_pac_pos.nelement() > 1:
                        print('need two corrections PACMAN')
                        print(imagined_image[:, leftover_pac_pos])
                        print(imagined_image[:, leftover_pac_pos][imagined_image[:, leftover_pac_pos] != PACMAN])
                        exit()



        #TODO: WHY THERE was NO p_curr_pos
        g_curr_pos = g_next_pos
        p_curr_pos = p_next_pos

    # COUNT GHOSTS
    g_samp_boot.append((sampled_obs_boot == GHOST).sum().item())
    g_samp_rpf.append((sampled_obs_rpf == GHOST).sum().item())
    g_compo_boot.append((compo_obs_boot == GHOST).sum().item())
    g_compo_rpf.append((compo_obs_rpf == GHOST).sum().item())
    g_vote_boot.append((vote_obs_boot == GHOST).sum().item())
    g_vote_rpf.append((vote_obs_rpf == GHOST).sum().item())
    # g_samp_boot.append((sampled_obs_boot == PACMAN).sum().item())
    # g_samp_rpf.append((sampled_obs_rpf == PACMAN).sum().item())
    # g_compo_boot.append((compo_obs_boot == PACMAN).sum().item())
    # g_compo_rpf.append((compo_obs_rpf == PACMAN).sum().item())
    # g_vote_boot.append((vote_obs_boot == PACMAN).sum().item())
    # g_vote_rpf.append((vote_obs_rpf == PACMAN).sum().item())

    # print(target_state.view((15,19))) # 0 - fruit, 6 (ghost), 1 (pcman), 5 (eaten cell)

    # Get Class accuracy (Fruit, Pacman, Ghost, Eaten cell)
    # They'll have nans whenever the selected class is not present in current reality
    acc_samp_boot_fruit_ec.append(
        (sampled_obs_boot[(target_state == FRUIT).nonzero()] == FRUIT).to(torch.float32).mean().item())
    acc_samp_rpf_fruit_ec.append(
        (sampled_obs_rpf[(target_state == FRUIT).nonzero()] == FRUIT).to(torch.float32).mean().item())
    acc_samp_boot_ghost_ec.append(
        (sampled_obs_boot[(target_state == GHOST).nonzero()] == GHOST).to(torch.float32).mean().item())
    acc_samp_rpf_ghost_ec.append(
        (sampled_obs_rpf[(target_state == GHOST).nonzero()] == GHOST).to(torch.float32).mean().item())
    acc_samp_boot_pacman_ec.append(
        (sampled_obs_boot[(target_state == PACMAN).nonzero()] == PACMAN).to(torch.float32).mean().item())
    acc_samp_rpf_pacman_ec.append(
        (sampled_obs_rpf[(target_state == PACMAN).nonzero()] == PACMAN).to(torch.float32).mean().item())
    acc_samp_boot_eaten_ec.append(
        (sampled_obs_boot[(target_state == EATEN_CELL).nonzero()] == EATEN_CELL).to(torch.float32).mean().item())
    acc_samp_rpf_eaten_ec.append(
        (sampled_obs_rpf[(target_state == EATEN_CELL).nonzero()] == EATEN_CELL).to(torch.float32).mean().item())

    acc_compo_boot_fruit_ec.append(
        (compo_obs_boot[(target_state == FRUIT).nonzero()] == FRUIT).to(torch.float32).mean().item())
    acc_compo_rpf_fruit_ec.append(
        (compo_obs_rpf[(target_state == FRUIT).nonzero()] == FRUIT).to(torch.float32).mean().item())
    acc_compo_boot_ghost_ec.append(
        (compo_obs_boot[(target_state == GHOST).nonzero()] == GHOST).to(torch.float32).mean().item())
    acc_compo_rpf_ghost_ec.append(
        (compo_obs_rpf[(target_state == GHOST).nonzero()] == GHOST).to(torch.float32).mean().item())
    acc_compo_boot_pacman_ec.append(
        (compo_obs_boot[(target_state == PACMAN).nonzero()] == PACMAN).to(torch.float32).mean().item())
    acc_compo_rpf_pacman_ec.append(
        (compo_obs_rpf[(target_state == PACMAN).nonzero()] == PACMAN).to(torch.float32).mean().item())
    acc_compo_boot_eaten_ec.append(
        (compo_obs_boot[(target_state == EATEN_CELL).nonzero()] == EATEN_CELL).to(torch.float32).mean().item())
    acc_compo_rpf_eaten_ec.append(
        (compo_obs_rpf[(target_state == EATEN_CELL).nonzero()] == EATEN_CELL).to(torch.float32).mean().item())

    acc_vote_boot_fruit_ec.append(
        (vote_obs_boot[(target_state == FRUIT).nonzero()] == FRUIT).to(torch.float32).mean().item())
    acc_vote_rpf_fruit_ec.append(
        (vote_obs_rpf[(target_state == FRUIT).nonzero()] == FRUIT).to(torch.float32).mean().item())
    acc_vote_boot_ghost_ec.append(
        (vote_obs_boot[(target_state == GHOST).nonzero()] == GHOST).to(torch.float32).mean().item())
    acc_vote_rpf_ghost_ec.append(
        (vote_obs_rpf[(target_state == GHOST).nonzero()] == GHOST).to(torch.float32).mean().item())
    acc_vote_boot_pacman_ec.append(
        (vote_obs_boot[(target_state == PACMAN).nonzero()] == PACMAN).to(torch.float32).mean().item())
    acc_vote_rpf_pacman_ec.append(
        (vote_obs_rpf[(target_state == PACMAN).nonzero()] == PACMAN).to(torch.float32).mean().item())
    acc_vote_boot_eaten_ec.append(
        (vote_obs_boot[(target_state == EATEN_CELL).nonzero()] == EATEN_CELL).to(torch.float32).mean().item())
    acc_vote_rpf_eaten_ec.append(
        (vote_obs_rpf[(target_state == EATEN_CELL).nonzero()] == EATEN_CELL).to(torch.float32).mean().item())














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
        print(np.bincount(g_samp_boot), np.bincount(g_samp_rpf), np.bincount(g_vote_boot), np.bincount(g_vote_rpf), np.bincount(g_compo_boot), np.bincount(g_compo_rpf))
        print(rewardsEpi)
        state = env.reset()
        state_torch = torch.FloatTensor(state).unsqueeze(0)

        g_curr_pos = env.get_ghost_pos()
        g_curr_pos = 15 * g_curr_pos[0] + g_curr_pos[1]
        p_curr_pos = env.get_pac_pos()
        p_curr_pos = 15 * p_curr_pos[0] + p_curr_pos[1]

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

        print('_ecUnified activation Boot: {} RPF: {}'.format(
            np.mean(np.array(acc_compo_boot_fruit_ec)[~np.isnan(np.array(acc_compo_boot_fruit_ec))]),
            np.mean(np.array(acc_compo_rpf_fruit_ec)[~np.isnan(np.array(acc_compo_rpf_fruit_ec))])
        ))
        print('_ecMajority vote Boot: {} RPF: {}'.format(
            np.mean(np.array(acc_vote_boot_fruit_ec)[~np.isnan(np.array(acc_vote_boot_fruit_ec))]),
            np.mean(np.array(acc_vote_rpf_fruit_ec)[~np.isnan(np.array(acc_vote_rpf_fruit_ec))])
        ))
        print('_ecSampling Boot: {} RPF: {}'.format(
            np.mean(np.array(acc_samp_boot_fruit_ec)[~np.isnan(np.array(acc_samp_boot_fruit_ec))]),
            np.mean(np.array(acc_samp_rpf_fruit_ec)[~np.isnan(np.array(acc_samp_rpf_fruit_ec))])
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
        print('_ecUnified activation Boot: {} RPF: {}'.format(
            np.mean(np.array(acc_compo_boot_ghost_ec)[~np.isnan(np.array(acc_compo_boot_ghost_ec))]),
            np.mean(np.array(acc_compo_rpf_ghost_ec)[~np.isnan(np.array(acc_compo_rpf_ghost_ec))])
        ))
        print('_ecMajority vote Boot: {} RPF: {}'.format(
            np.mean(np.array(acc_vote_boot_ghost_ec)[~np.isnan(np.array(acc_vote_boot_ghost_ec))]),
            np.mean(np.array(acc_vote_rpf_ghost_ec)[~np.isnan(np.array(acc_vote_rpf_ghost_ec))])
        ))
        print('_ecSampling Boot: {} RPF: {}'.format(
            np.mean(np.array(acc_samp_boot_ghost_ec)[~np.isnan(np.array(acc_samp_boot_ghost_ec))]),
            np.mean(np.array(acc_samp_rpf_ghost_ec)[~np.isnan(np.array(acc_samp_rpf_ghost_ec))])
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
        print('_ecUnified activation Boot: {} RPF: {}'.format(
            np.mean(np.array(acc_compo_boot_pacman_ec)[~np.isnan(np.array(acc_compo_boot_pacman_ec))]),
            np.mean(np.array(acc_compo_rpf_pacman_ec)[~np.isnan(np.array(acc_compo_rpf_pacman_ec))])
        ))
        print('_ecMajority vote Boot: {} RPF: {}'.format(
            np.mean(np.array(acc_vote_boot_pacman_ec)[~np.isnan(np.array(acc_vote_boot_pacman_ec))]),
            np.mean(np.array(acc_vote_rpf_pacman_ec)[~np.isnan(np.array(acc_vote_rpf_pacman_ec))])
        ))
        print('_ecSampling Boot: {} RPF: {}'.format(
            np.mean(np.array(acc_samp_boot_pacman_ec)[~np.isnan(np.array(acc_samp_boot_pacman_ec))]),
            np.mean(np.array(acc_samp_rpf_pacman_ec)[~np.isnan(np.array(acc_samp_rpf_pacman_ec))])
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
        print('_ecUnified activation Boot: {} RPF: {}'.format(
            np.mean(np.array(acc_compo_boot_eaten_ec)[~np.isnan(np.array(acc_compo_boot_eaten_ec))]),
            np.mean(np.array(acc_compo_rpf_eaten_ec)[~np.isnan(np.array(acc_compo_rpf_eaten_ec))])
        ))
        print('_ecMajority vote Boot: {} RPF: {}'.format(
            np.mean(np.array(acc_vote_boot_eaten_ec)[~np.isnan(np.array(acc_vote_boot_eaten_ec))]),
            np.mean(np.array(acc_vote_rpf_eaten_ec)[~np.isnan(np.array(acc_vote_rpf_eaten_ec))])
        ))
        print('_ecSampling Boot: {} RPF: {}'.format(
            np.mean(np.array(acc_samp_boot_eaten_ec)[~np.isnan(np.array(acc_samp_boot_eaten_ec))]),
            np.mean(np.array(acc_samp_rpf_eaten_ec)[~np.isnan(np.array(acc_samp_rpf_eaten_ec))])
        ))
        print("===============================================================================")


    # For error correction
    old_state_cat = target_state.clone()

