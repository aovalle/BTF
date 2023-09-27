'''
4 march 2020
this includes the 2nd wave of EC. It consists of dealing with additional ghosts/pacmans
'''
import sys
sys.path.append('../')
import os
#from utils.misc import saveImg
from params import Params as p
import numpy as np
import torch
import torch.nn.functional as F
import copy

from error_correction import correct_missing, correct_additional
import minipacman.minipacman_utils as mpu

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

class RollingHorizon():

    def __init__(self, input_shape, num_actions, learned_env_model=None, mhead=False):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.env_model = learned_env_model
        self.rollout_batch_size = 1
        self.curr_rollout = None
        self.simulator = None
        self.mhead = mhead
        self.pos = None

        self.trajecImg = []
        self.trajecECImg = []
        self.trajecRew = []

    def select_action(self, obs, simulator=None, cloned_state=None, comp_state=None, saveDir=None, type=None, ec_pos=None):
        if p.ERROR_CORRECTION:
            self.pos = ec_pos

        if obs.ndim == 4:
            obs = torch.FloatTensor(obs)#.to(device)                 # If it's already batcherized
        else:
            obs = torch.FloatTensor(obs).unsqueeze(0)#.to(device)    # Batcherize it so it has batch X channel X w X h

        if self.curr_rollout is None or not p.SHIFT_BUFFER:
            # Generate random rollout
            rollout = np.random.randint(self.num_actions, size=p.SEQ_LEN)
        else: # if we want to use shift buffer
            rollout = self.shift_buffer(np.copy(self.curr_rollout))
        #print('Initial rollout ', rollout)

        # Evaluate rollouts and get the best one
        if self.env_model is not None:
            if not self.mhead:
                self.curr_rollout = self.eval_seq_img(obs, rollout, saveDir)
            else:
                self.curr_rollout = self.eval_seq_img_mhead(obs, rollout, saveDir, type)

        else:   # Perfect simulator
            self.curr_rollout = self.eval_seq(cloned_state, rollout, simulator, comp_state, saveDir)
        action = self.curr_rollout[0]
        return action, self.curr_rollout



    def testBestRoll(self, obs, simulator, init_state, roll):
        cloned_env = copy.deepcopy(simulator)
        # (Re)init simulated environment
        env = copy.deepcopy(cloned_env)
        # print(np.array_equal(env.env.world_state['food'], comp_state['food']))
        env.env.world_state = copy.deepcopy(init_state)

        print(roll)
        print('GHOST SIM ', env.env.get_ghost_pos())
        print('PAC SIM ', env.env.get_pac_pos())
        rollout_reward=0
        for step in range(p.SEQ_LEN):
            action = roll[step]  # 1
            state, reward, done, _ = env.step(action)
            rollout_reward += reward

            saveImg(state, 'samples-rh/test-sim/', join('simstate' + str(step) + '.png'))
            print('GHOST SIM ', env.env.get_ghost_pos())
            print('PAC SIM ', env.env.get_pac_pos())

            if done:
                break




    def eval_seq(self, init_state, rollout, simulator, comp_state, saveDir):
        cloned_env = copy.deepcopy(simulator)

        highest_reward = float("-inf")
        best_rollout = None
        for e in range(p.N_EVALS):
            if e == 0:
                mut_rollout = rollout
            else:
                mut_rollout = self.mutate(np.copy(rollout))
            #print('Mutated rollout ', mut_rollout)
            # todo: maybe i have to reinstante the whole env
            # for example what about the points and stuff

            # (Re)init simulated environment
            env = copy.deepcopy(cloned_env)
            #print(np.array_equal(env.env.world_state['food'], comp_state['food']))
            env.env.world_state = copy.deepcopy(init_state)

            rollout_reward = 0
            for step in range(p.SEQ_LEN):
                #print(mut_rollout)
                action = mut_rollout[step] #1
                #print(action)
                state, reward, done, _ = env.step(action)
                rollout_reward += reward

                saveImg(state, 'samples-rh/simulated/', join('simstate'+str(step)+'eval'+str(e)+'.png'))

                if done:
                    break
            if rollout_reward > highest_reward:
                best_rollout = np.copy(mut_rollout)
                highest_reward = rollout_reward
        return best_rollout


    def eval_seq_img(self, init_state, rollout, saveDir):

        highest_reward = float("-inf")
        best_rollout = None
        for e in range(p.N_EVALS):
            if e == 0:
                mut_rollout = rollout
            else:
                mut_rollout = self.mutate(np.copy(rollout))

            state = init_state
            rollout_reward = 0
            for step in range(p.SEQ_LEN):
                action = mut_rollout[step] #1

                # Create (onehot) action grids. Shape (batch, #actions, w, h)
                onehot_action = torch.zeros(self.rollout_batch_size, self.num_actions, *self.input_shape[1:])
                # for the first instance of the batch action 0 is onehot, for the second instance action 1 is onehot, and so on
                onehot_action[range(self.rollout_batch_size), action] = 1
                state_action = torch.cat([state, onehot_action], 1).to(device)


                # get imagined states and rewards (for all actions) (Shapes: state:(285*sizebatch, #pixel types) - rew:(sizebatch, #rew types))
                imagined_state, imagined_reward = self.env_model.imagine(state_action)
                #print(init_state.shape, imagined_state.shape)

                # First clone im state to start to prepare it for graphical mode
                imagined_image = F.softmax(imagined_state, dim=1)

                #imagined_image = mpu.target_to_pix(imagined_state.view(self.rollout_batch_size, -1, len(mpu.pixels))[0].max(1)[1].data.cpu().numpy())
                imagined_image = mpu.target_to_pix(imagined_image.view(self.rollout_batch_size, -1, len(mpu.pixels))[0].max(1)[1].data.cpu().numpy())
                imagined_image = torch.FloatTensor(imagined_image.reshape(15, 19, 3)).unsqueeze(0)
                # If i save it in something else that is not torch I SHOULD NOT DO THIS
                imagined_image = imagined_image.permute(0, 3, 1, 2)

                imagined_reward = F.softmax(imagined_reward, dim=1).max(1)[1].cpu()
                imagined_reward = mpu.mode_rewards['regular'][imagined_reward]
                rollout_reward += imagined_reward#.item()

                saveImg(imagined_image, saveDir, join('simstate' + str(step) + 'eval' + str(e) + '.png'))

                #note: how to consider the dones?

                # Get next IMAGINARY actions based on IMAGINARY states to continue IMAGINARY rollouts
                state = imagined_image #imagined_state

            if rollout_reward > highest_reward:
                best_rollout = np.copy(mut_rollout)
                highest_reward = rollout_reward

        #print('Best rollout ', best_rollout, ' highest reward ', highest_reward)
        return best_rollout

    def correction(self, pred_image, headed_pred_image, mode, previous_state):
        GHOST = 6
        PACMAN = 1
        # Collapse activations into heads
        headed_pred_image = F.softmax(headed_pred_image, dim=2).max(2)[1]


        if (pred_image == GHOST).sum().item() == 0:
            # First check that if there is no ghost predicted, maybe it's because
            # the ghost is dead (or yellow due to fruit) in that case don't EC
            # So check against last known real obs
            if (previous_state == GHOST).sum().item() > 0:
                pred_image, self.pos['g'] = correct_missing(pred_image, headed_pred_image, p.GHOST, mode, self.pos['cg'], True)
                self.pos['cg'] = self.pos['g']
        # If there was more than one ghost
        elif (pred_image == GHOST).sum().item() > 1:
            pred_image, self.pos['g'] = correct_additional(pred_image, headed_pred_image, p.GHOST, mode, previous_state, True)
            self.pos['cg'] = self.pos['g']

        if (pred_image == PACMAN).sum().item() == 0:
            pred_image, self.pos['p'] = correct_missing(pred_image, headed_pred_image, p.PACMAN, mode, self.pos['cp'], True)
            self.pos['cp'] = self.pos['p']
        # Check if there's more than one pacman in the frame
        elif (pred_image == PACMAN).sum().item() > 1:
            pred_image, self.pos['p'] = correct_additional(pred_image, headed_pred_image, p.PACMAN, mode, previous_state, True)
            self.pos['cp'] = self.pos['p']

        # # FIXME: IS THIS CORRECT? OR THEY SHOULD BE INSIDE THE IFS
        # if self.pos['g'] is None or self.pos['p'] is None:
        #     print('NONE')
        #     print(self.pos['g'], self.pos['p'])
        #     exit()
        # self.pos['cg'] = self.pos['g']
        # self.pos['cp'] = self.pos['p']
        return pred_image

    def eval_seq_img_mhead(self, init_state, rollout, saveDir, type):

        trajDir = join('samples-trajec/')
        self.trajecECImg = []
        self.trajecImg = []
        self.trajecRew = []

        highest_reward = float("-inf")
        best_rollout = None
        for e in range(p.N_EVALS):
            if e == 0:
                mut_rollout = rollout
            else:
                mut_rollout = self.mutate(np.copy(rollout))

            tempTrRew = []
            state = init_state
            rollout_reward = 0

            # this is the most recent real observation
            if p.ERROR_CORRECTION:
                previous_state = init_state.cpu().numpy()
                previous_state = torch.LongTensor(mpu.pix_to_target(previous_state)).to(device)

            for step in range(p.SEQ_LEN):
                action = mut_rollout[step] #1

                # Create (onehot) action grids. Shape (batch, #actions, w, h)
                onehot_action = torch.zeros(self.rollout_batch_size, self.num_actions, *self.input_shape[1:])
                # for the first instance of the batch action 0 is onehot, for the second instance action 1 is onehot, and so on
                onehot_action[range(self.rollout_batch_size), action] = 1
                state_action = torch.cat([state, onehot_action], 1).to(device)


                # get imagined states and rewards (for all actions) (Shapes: state:(285*sizebatch, #pixel types) - rew:(sizebatch, #rew types))
                imagined_state, imagined_reward = self.env_model.imagine(state_action, list(range(p.NUM_HEADS)))
                imagined_state = torch.stack(imagined_state)
                imagined_reward = torch.stack(imagined_reward)

                # Copy predictions for type of composite
                pred_state = torch.clone(imagined_state)
                pred_reward = torch.clone(imagined_reward)

                if 'avg' in type:
                    # Avg activations
                    avg_pred_state = torch.mean(pred_state, dim=0)
                    avg_pred_reward = torch.mean(pred_reward, dim=0)
                    # Get predicted classes
                    pred_reward = F.softmax(avg_pred_reward, dim=1).max(1)[1].cpu().item()
                    avg_pred_state = F.softmax(avg_pred_state, dim=1).max(1)[1]#.cpu()
                    #avg_pred_state = F.softmax(avg_pred_state, dim=1)#.max(1)[1].cpu()

                    pred_reward = mpu.mode_rewards["regular"][pred_reward]
                    rollout_reward += pred_reward
                    if p.ERROR_CORRECTION:
                        ec_image = self.correction(avg_pred_state, pred_state, 'compo', previous_state)
                        self.trajecECImg.append(join(trajDir + 'simstateEC' + str(step) + 'eval' + str(e) + '.png'))
                        prepAndSaveImg(ec_image, trajDir, join('simstateEC' + str(step) + 'eval' + str(e) + '.png'))
                        imagined_imageEC = mpu.target_to_pix(ec_image.data.cpu().numpy())
                    #imagined_image = mpu.target_to_pix(avg_pred_state.view(self.rollout_batch_size, -1, len(mpu.pixels))[0].max(1)[1].data.cpu().numpy())
                    imagined_image = mpu.target_to_pix(avg_pred_state.data.cpu().numpy())
                elif 'vote' in type:
                    # Get predicted classes
                    vote_pred_reward = F.softmax(pred_reward, dim=2).max(2)[1]#.cpu().item()
                    vote_pred_state = F.softmax(pred_state, dim=2).max(2)[1]#.cpu()
                    # Get majority vote
                    #print(vote_pred_reward)
                    vote_pred_state = torch.mode(vote_pred_state, dim=0)[0]
                    pred_reward = torch.mode(vote_pred_reward, dim=0)[0].cpu().item()

                    pred_reward = mpu.mode_rewards["regular"][pred_reward]
                    rollout_reward += pred_reward
                    if p.ERROR_CORRECTION:
                        ec_image = self.correction(vote_pred_state, pred_state, 'vote', previous_state)
                        self.trajecECImg.append(join(trajDir + 'simstateEC' + str(step) + 'eval' + str(e) + '.png'))
                        prepAndSaveImg(ec_image, trajDir, join('simstateEC' + str(step) + 'eval' + str(e) + '.png'))
                        imagined_imageEC = mpu.target_to_pix(ec_image.data.cpu().numpy())
                    imagined_image = mpu.target_to_pix(vote_pred_state.data.cpu().numpy())

                elif 'sample' in type:
                    # Get predicted classes
                    samp_pred_reward = F.softmax(pred_reward, dim=2).max(2)[1]
                    samp_pred_state = F.softmax(pred_state, dim=2).max(2)[1]
                    samp_pred_state = torch.tensor(
                        np.choose(np.random.randint(samp_pred_state.shape[0], size=samp_pred_state.shape[1]),
                                  samp_pred_state.cpu().numpy())).to(device)

                    pred_reward = np.choose(np.random.randint(samp_pred_reward.shape[0], size=samp_pred_reward.shape[1]),
                                  samp_pred_reward.cpu().numpy()).item()

                    pred_reward = mpu.mode_rewards["regular"][pred_reward]
                    rollout_reward += pred_reward
                    if p.ERROR_CORRECTION:
                        ec_image = self.correction(samp_pred_state, pred_state, 'sampled', previous_state)
                        self.trajecECImg.append(join(trajDir+'simstateEC'+str(step)+'eval'+str(e)+'.png'))
                        prepAndSaveImg(ec_image, trajDir, join('simstateEC'+str(step)+'eval'+str(e)+'.png'))
                        imagined_imageEC = mpu.target_to_pix(ec_image.data.cpu().numpy())
                    imagined_image = mpu.target_to_pix(samp_pred_state.data.cpu().numpy())

                imagined_image = torch.FloatTensor(imagined_image.reshape(15, 19, 3)).unsqueeze(0)
                # If i save it in something else that is not torch I SHOULD NOT DO THIS
                imagined_image = imagined_image.permute(0, 3, 1, 2)

                if p.ERROR_CORRECTION:
                    imagined_imageEC = torch.FloatTensor(imagined_imageEC.reshape(15, 19, 3)).unsqueeze(0)
                    # If i save it in something else that is not torch I SHOULD NOT DO THIS
                    imagined_imageEC = imagined_imageEC.permute(0, 3, 1, 2)

                # Save non-EC trajectories
                self.trajecImg.append(join(saveDir+type+'simstate'+str(step)+'eval'+str(e)+'.png'))
                saveImg(imagined_image, saveDir, join(type + 'simstate' + str(step) + 'eval' + str(e) + '.png'))

                # Save predicted rewards in rollout
                tempTrRew.append(pred_reward)

                # Get next IMAGINARY actions based on IMAGINARY states to continue IMAGINARY rollouts
                if p.ERROR_CORRECTION:
                    state = imagined_imageEC #imagined_state
                    previous_state = ec_image
                else:
                    state = imagined_image

                if pred_reward == -1:
                    break

            self.trajecRew.append(tempTrRew)

            if rollout_reward > highest_reward:
                best_rollout = np.copy(mut_rollout)
                highest_reward = rollout_reward

        #print('Best rollout ', best_rollout, ' highest reward ', highest_reward)
        return best_rollout


    def mutate(self, rollout):
        #rollout[np.random.rand(*rollout.shape) < p.MUT_RATE] = 99
        # Determine how many elements to sample from the rollout
        n = np.random.binomial(len(rollout), p.MUT_RATE)
        # Generate indices of the sequence to be mutated
        idx = d = np.unique(np.random.randint(0, len(rollout), size=n))
        # Sample new actions
        new_actions = np.random.randint(self.num_actions, size=len(d))
        # Substitute values and obtain mutated sequence
        rollout[idx] = new_actions
        return rollout

    def shift_buffer(self, rollout):
        # append new random action at the end
        sf_rollout = np.append(rollout, np.random.randint(self.num_actions))
        # remove first action
        sf_rollout = np.delete(sf_rollout, 0)
        return sf_rollout


