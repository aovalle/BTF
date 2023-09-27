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
import PIL

import os
from os.path import join
from torchvision.utils import save_image
import torch


#from a2c import A2C
from models.env_model_boot import EnvModel


def processImgToSave(image):
    # Transform 1-D 285 pixel vector from its categorical form into pixel-channel form (285 x 3)
    image = mpu.target_to_pix(image.cpu().numpy())
    # reshape into image dimensions (1 x 15 x 19 x 3)
    image = torch.FloatTensor(image.reshape(15, 19, 3)).unsqueeze(0)
    image = image.permute(0, 3, 1, 2)
    return image

def saveImg(image, dir, fname):
    img = np.copy(image)
    state_image = torch.FloatTensor(img).unsqueeze(0)  # .permute(1, 2, 0).cpu().numpy()
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
env_model = EnvModel(input_space, mpu.num_pixels, mpu.num_rewards, p.NUM_HEADS)
env_model.load_state_dict('worldModels/env_model_boot_rand_25k')

# Init Rolling Horizon agent
agentRH = RollingHorizon(input_space, action_space, env_model, mhead=True)


types = ['boot+avg', 'boot+vote', 'boot+sample']
type = types[2]

epiRewards = 0
rewards = []
steps = []
last_t = 0
episode = 0
simDir = join('samples-rh+model+' + type +'/simulated/') # Avg activations
realDir = join('samples-rh+model+' + type + '/real/') # Avg activations
trajDir = join('samples-trajec+model+'+type+'/')
saveImg(state, realDir, join('ep' + str(episode) + 'init.png'))

# ERROR CORRECTION
pos = {'p':None, 'cp':None, 'g':None, 'cg':None}
g_curr_pos = env.get_ghost_pos()
pos['cg'] = 15*g_curr_pos[0] + g_curr_pos[1]
p_curr_pos = env.get_pac_pos()
pos['cp'] = 15*p_curr_pos[0] + p_curr_pos[1]

trajecImg = []
trajecECImg = []
for step in range(p.STEPS):
    # Seqs are:
    # s, s', is', is'' ... is'''''''

    # save current state
    saveImg(state, trajDir, 'rstate.png')

    # Pass current state and a perfect copy of the environment
    # and get an action sequence
    action_seq = agentRH.select_action(state, saveDir=simDir, type=type, ec_pos=pos)
    state, reward, done, _ = env.step(action_seq[0])
    epiRewards += reward

    # Save new state
    saveImg(state, trajDir, 'rnstate.png')

    if p.ERROR_CORRECTION:
        for e in range(p.N_EVALS):
            trajecImg.append(join(trajDir + 'rstate.png'))
            trajecECImg.append(join(trajDir + 'rstate.png'))
            trajecImg.append(join(trajDir + 'rnstate.png'))
            trajecECImg.append(join(trajDir + 'rnstate.png'))

            #Get only those imagined frames corresponding to aspecific evaluation
            trajecImg += [tr for tr in agentRH.trajecImg if 'eval'+str(e) in tr]
            trajecECImg += [tr for tr in agentRH.trajecECImg if 'eval'+str(e) in tr]

            # Save trajec photos
            imgs = [PIL.Image.open(i) for i in trajecImg]
            imgsEC = [PIL.Image.open(i) for i in trajecECImg]
            # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
            min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
            imgs_comb = np.hstack([np.asarray(i.resize(min_shape)) for i in imgs])
            imgs_combEC = np.hstack([np.asarray(i.resize(min_shape)) for i in imgsEC])
            iim = np.vstack((imgs_comb, imgs_combEC))
            iim = PIL.Image.fromarray(iim)

            iim.save(join('samples-trajec+model+'+type+'/trajEpi'+str(episode)+
                          'Eval'+str(e)+'Step'+ str(step)+'-Rews'+
                          '-'.join(str(i) for i in agentRH.trajecRew[e])+'.png'))

            trajecImg = []
            trajecECImg = []

    if done:
        saveImg(state, realDir, join('ep' + str(episode) + 'end.png'))
        state = env.reset()
        rewards.append(epiRewards)
        steps.append(step-last_t)
        epiRewards = 0
        print('Episode {} Steps {} Rewards {}'.format(episode, step-last_t, rewards[-1]))
        last_t = step
        episode += 1
        saveImg(state, realDir, join('ep' + str(episode) + 'init.png'))
        agentRH.curr_rollout = None

        if episode > 9:
            break

print('Reward Average {} Std Dev {}'.format(np.mean(rewards), np.std(rewards)))
print('Step Average {} Std Dev {}'.format(np.mean(steps), np.std(steps)))