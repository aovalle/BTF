import numpy as np
from minipacman.minipacman import MiniPacman
import matplotlib.pyplot as plt
import seaborn as sns
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join
from torchvision.utils import save_image
from env_model import EnvModel#, EnvModelNonProb
import minipacman.minipacman_utils as mpu
from scipy.stats import entropy
from params import Params as p

def displayImage(image, step, reward):
    s = "step" + str(step) + " reward " + str(reward)
    plt.title(s)
    plt.imshow(image)
    plt.show()

def processImgToSave(image):
    # Transform 1-D 285 pixel vector from its categorical form into pixel-channel form (285 x 3)
    image = mpu.target_to_pix(image.cpu().numpy())
    # reshape into image dimensions (1 x 15 x 19 x 3)
    image = torch.FloatTensor(image.reshape(15, 19, 3)).unsqueeze(0)
    image = image.permute(0, 3, 1, 2)
    return image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

keys = {'w': 2,'d': 1,'a': 3,'s': 4,'z': 0, 'q':-1}

MODES = ('regular', 'avoid', 'hunt', 'ambush', 'rush')
frame_cap = 1000

# Set up env
mode = 'regular'
env = MiniPacman(mode, 1000)

# Init env model
input_space = env.observation_space.shape
action_space = env.action_space.n
env_model = EnvModel(input_space, mpu.num_pixels, mpu.num_rewards, p.NUM_HEADS)
env_model_rpf = EnvModel(input_space, mpu.num_pixels, mpu.num_rewards, p.NUM_HEADS)
#env_model_a2c = EnvModelNonProb(input_space, mpu.num_pixels, mpu.num_rewards)
fname = 'env_model_boot_rew_a2c'
env_model.load_state_dict(fname, rpf=False)
fname = 'env_model_rpf_rew_a2c'
env_model_rpf.load_state_dict(fname)
# fname = 'env_model_a2c_rew_a2c'
# env_model_a2c.load_state_dict(fname)

# Numpy array - state shape (3,15,19)
done = False
state = env.reset()

total_reward = 0
step = 0

# Green - pacman, Red - ghost
# State original is: ch X w X h (3,15,19) transformed for displaying into: w X h X c
displayImage(state.transpose(1, 2, 0), step, total_reward)

state_torch = torch.FloatTensor(state).unsqueeze(0)
if not os.path.isdir('samples/real'):
    os.makedirs('samples/real/')
save_image(state_torch, 'samples/real/init.png')

total_err_head_rpf = []
total_err_head_boot = []
total_err_composite_boot = []
total_err_composite_rpf = []
total_acc_vote_boot = []
total_acc_vote_rpf = []
total_acc_sampled_boot = []
total_acc_sampled_rpf = []

while not done:
    plt.clf()

    # Get action from a human
    print("Write input")
    x = input()
    print(x)
    try:
        keys[x]
    except:
        print("Only 'w' 'a' 'd' 's'") # a - left, d - right, w - up, s - down
        continue
    action = keys[x]

    if action == -1:
        break

    # Create inputs for env model
    # Create (onehot) action grids. Shape (batch, #actions, w, h)
    onehot_action = torch.zeros(1, action_space, *input_space[1:])
    onehot_action[0, action] = 1
    inputs = torch.cat([state_torch, onehot_action], 1).to(device)
    # Send them and retrieve what the env model heads think it'll happen
    imagined_obs, imagined_reward = env_model.imagine(inputs, list(range(p.NUM_HEADS)), rpf=False)
    imagined_obs_rpf, imagined_reward_rpf = env_model_rpf.imagine(inputs, list(range(p.NUM_HEADS)))

    imagined_obs = torch.stack(imagined_obs)
    imagined_reward = torch.stack(imagined_reward)
    imagined_obs_rpf = torch.stack(imagined_obs_rpf)
    imagined_reward_rpf = torch.stack(imagined_reward_rpf)

    # Clone prediction for future error comparison
    pred_obs_boot, pred_rew_boot = torch.clone(imagined_obs), torch.clone(imagined_reward)
    pred_obs_rpf, pred_rew_rpf = torch.clone(imagined_obs_rpf), torch.clone(imagined_reward_rpf)

    print(imagined_obs.shape, pred_obs_boot.shape)
    print(imagined_reward.shape)

    # Dim: heads X outputs (1, we're predicting one reward) X type of rewards
    # Get the maximum in dim 2 (type of reward) but transform it into softmax and get the index
    imagined_reward = F.softmax(imagined_reward, dim=2).max(2)[1].cpu()
    imagined_reward = imagined_reward.squeeze(1).data.numpy()
    imagined_reward_rpf = F.softmax(imagined_reward_rpf, dim=2).max(2)[1].cpu()
    imagined_reward_rpf = imagined_reward_rpf.squeeze(1).data.numpy()


    # Calculate entropy
    value, counts = np.unique(imagined_reward, return_counts=True)
    H = entropy(counts)
    value_rpf, counts_rpf = np.unique(imagined_reward_rpf, return_counts=True)
    H_rpf = entropy(counts_rpf)

    #ax = plt.axes()
    ax = sns.distplot(imagined_reward, bins=range(0, 9 + 1, 1), kde=False)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.5)
    ax.set_title('Reward prediction distribution \n Entropy: {:.2f}'.format(H), fontsize=15)
    ax.set(xlabel='Predicted reward', ylabel='Number of heads')
    # plt.show()
    if not os.path.isdir('samples-boot/entrew'):
        os.makedirs('samples-boot/entrew')
    plt.savefig(join('samples-boot/entrew/entrew' + str(step) + '.png'), bbox_inches='tight', pad_inches=0)
    plt.clf()

    ax = sns.distplot(imagined_reward_rpf, bins=range(0, 9 + 1, 1), kde=False)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.5)
    ax.set_title('Reward prediction distribution \n Entropy: {:.2f}'.format(H_rpf), fontsize=15)
    ax.set(xlabel='Predicted reward', ylabel='Number of heads')
    # plt.show()
    if not os.path.isdir('samples-rpf/entrew'):
        os.makedirs('samples-rpf/entrew')
    plt.savefig(join('samples-rpf/entrew/entrew' + str(step) + '.png'), bbox_inches='tight', pad_inches=0)
    plt.clf()

    # Dim: heads X outputs (285, we're predicting 285 pixels) X type of pix
    # First clone im state to start to prepare it for graphical mode
    print('pre softmax ', imagined_obs.shape)
    imagined_image = F.softmax(imagined_obs, dim=2).max(2)[1]
    print('post softmax ', imagined_image.shape)

    # Get entropy per pixel
    H_pix = []
    for i in range(imagined_image.shape[1]):
        # Apply for each pixel
        value, counts = np.unique(imagined_image[:, i].cpu(), return_counts=True)
        H_pix.append(entropy(counts))

    # Dim: heads X outputs (285, we're predicting 285 pixels) X type of pix
    # First clone im state to start to prepare it for graphical mode
    imagined_image_rpf = F.softmax(imagined_obs_rpf, dim=2).max(2)[1]
    # Get entropy per pixel
    H_pix_rpf = []
    for i in range(imagined_image_rpf.shape[1]):
        # Apply for each pixel
        value_rpf, counts_rpf = np.unique(imagined_image_rpf[:, i].cpu(), return_counts=True)
        H_pix_rpf.append(entropy(counts_rpf))


    #imagined_images = []
    # Render imagined observations for visualization
    for head in range(p.NUM_HEADS):
        # # Transform 1-D 285 pixel vector from its categorical form into pixel-channel form (285 x 3)
        # imagined_head_image = mpu.target_to_pix(imagined_image[head, :].cpu().numpy())
        # # reshape into image dimensions (1 x 15 x 19 x 3)
        # imagined_head_image = torch.FloatTensor(imagined_head_image.reshape(15, 19, 3)).unsqueeze(0)
        # imagined_head_image = imagined_head_image.permute(0, 3, 1, 2)
        imagined_head_image = processImgToSave(imagined_image[head, :])
        if not os.path.isdir( join('samples-boot/head' + str(head)) ):
            os.makedirs(join('samples-boot/head' + str(head)))
        save_image(imagined_head_image, join('samples-boot/head' + str(head) + '/' + str(head) + 'step' + str(step) + '.png'))
    # Render pixel entropy heatmap every step
    H_pix = np.array(H_pix).reshape(15, 19)

    # imagined_images = []
    # Render imagined observations for visualization
    for head in range(p.NUM_HEADS):
        # Transform 1-D 285 pixel vector from its categorical form into pixel-channel form (285 x 3)
        # imagined_head_image_rpf = mpu.target_to_pix(imagined_image_rpf[head, :].cpu().numpy())
        # # reshape into image dimensions (1 x 15 x 19 x 3)
        # imagined_head_image_rpf = torch.FloatTensor(imagined_head_image_rpf.reshape(15, 19, 3)).unsqueeze(0)
        # imagined_head_image_rpf = imagined_head_image_rpf.permute(0, 3, 1, 2)
        imagined_head_image_rpf = processImgToSave(imagined_image[head, :])
        if not os.path.isdir( join('samples-rpf/head' + str(head)) ):
            os.makedirs(join('samples-rpf/head' + str(head)))
        save_image(imagined_head_image_rpf,
                   join('samples-rpf/head' + str(head) + '/' + str(head) + 'step' + str(step) + '.png'))
    # Render pixel entropy heatmap every step
    H_pix_rpf = np.array(H_pix_rpf).reshape(15, 19)


    # Generate Graphical grid for cell entropy (observations
    cmap = sns.cm.rocket_r
    ax = sns.heatmap(H_pix, linewidth=0.5, cmap=cmap, vmax=2.31) #cmap="Blues") #, annot=True) #cbar=False #vmax=max entropy
    #sns.heatmap(H_pix, ax=ax, linewidth=0.5, cmap=cmap, vmax=2.31) #cmap="Blues") #, annot=True) #cbar=False #vmax=max entropy
    ax.set_title('Cell entropy \n Cumulative entropy: {:.2f}'.format(np.sum(H_pix)), fontsize=15)
    #ax.set_ylim(-0.5, 15)
    ax.set_ylim(15, -0.5)
    # plt.imshow(H_pix, cmap='hot', interpolation='nearest')
    if not os.path.isdir('samples-boot/entropy'):
        os.makedirs('samples-boot/entropy')
    plt.savefig(join('samples-boot/entropy/entropy' + str(step) + '.png'), bbox_inches='tight', pad_inches=0)
    plt.clf()


    # cmap = sns.cubehelix_palette(8, start=.5, rot=-.75)
    cmap = sns.cm.rocket_r
    ax = sns.heatmap(H_pix_rpf, linewidth=0.5, cmap=cmap,
                     vmax=2.31)  # cmap="Blues") #, annot=True) #cbar=False #vmax=max entropy
    # sns.heatmap(H_pix, ax=ax, linewidth=0.5, cmap=cmap, vmax=2.31) #cmap="Blues") #, annot=True) #cbar=False #vmax=max entropy
    ax.set_title('Cell entropy \n Cumulative entropy: {:.2f}'.format(np.sum(H_pix_rpf)), fontsize=15)
    # ax.set_ylim(-0.5, 15)
    ax.set_ylim(15, -0.5)
    # plt.imshow(H_pix, cmap='hot', interpolation='nearest')
    if not os.path.isdir('samples-rpf/entropy'):
        os.makedirs('samples-rpf/entropy')
    plt.savefig(join('samples-rpf/entropy/entropy' + str(step) + '.png'), bbox_inches='tight', pad_inches=0)
    plt.clf()


    # Get actual transition
    next_state, reward, done, _ = env.step(action)

    print('Reward {} Imagined reward {} Entropy Reward {} Entropy observation {} action {}'.format(reward, imagined_reward, H, np.sum(H_pix), action))
    print('Reward {} Imagined reward {} Entropy Reward {} Entropy observation {} action {}'.format(reward,
                                                                                                   imagined_reward_rpf, H_rpf,
                                                                                                   np.sum(H_pix_rpf),
                                                                                                   action))

    state_torch = torch.FloatTensor(next_state).unsqueeze(0)
    if not os.path.isdir('samples-boot/real'):
        os.makedirs('samples-boot/real')
    if not os.path.isdir('samples-rpf/real'):
        os.makedirs('samples-rpf/real')
    save_image(state_torch, join('samples-boot/real/real' + str(step) + '.png'))
    save_image(state_torch, join('samples-rpf/real/real' + str(step) + '.png'))


    total_reward += reward
    displayImage(next_state.transpose(1, 2, 0), step, total_reward)
    step += 1


    #Compare next_state and reward
    #note four approaches: i can avg the activations, i can vote, i can sample or i can avg heads errors wrt to targets
    #target_state = np.copy(next_state)
    #print(target_state)
    target_state = np.expand_dims(next_state, axis=0) if next_state.ndim != 4 else next_state
    criterion = nn.CrossEntropyLoss()
    target_state = torch.LongTensor(mpu.pix_to_target(target_state)).to(device)

    # Get individual head errors then avg the errors
    err_head_rpf, err_head_boot = [], []
    for h in range(p.NUM_HEADS):
        err_head_boot.append(criterion(pred_obs_boot[h], target_state))
        err_head_rpf.append(criterion(pred_obs_rpf[h], target_state))
    total_err_head_rpf.append(torch.stack(err_head_rpf).mean())
    total_err_head_boot.append(torch.stack(err_head_boot).mean())

    # Composite (Avg activations across heads to obtain a unified output and unified error)
    compo_obs_boot = torch.mean(pred_obs_boot, dim=0)
    compo_obs_rpf = torch.mean(pred_obs_rpf, dim=0)
    total_err_composite_boot.append(criterion(compo_obs_boot, target_state))
    total_err_composite_rpf.append(criterion(compo_obs_rpf, target_state))
    compo_obs_boot = F.softmax(compo_obs_boot, dim=1).max(1)[1]
    compo_obs_boot = processImgToSave(compo_obs_boot)
    if not os.path.isdir(join('samples-boot/chosen-future')):
        os.makedirs(join('samples-boot/chosen-future'))
    save_image(compo_obs_boot,
               join('samples-boot/chosen-future/composite-step' + str(step) + '.png'))
    compo_obs_rpf = F.softmax(compo_obs_rpf, dim=1).max(1)[1]
    compo_obs_rpf = processImgToSave(compo_obs_rpf)
    if not os.path.isdir(join('samples-rpf/chosen-future')):
        os.makedirs(join('samples-rpf/chosen-future'))
    save_image(compo_obs_rpf,
               join('samples-rpf/chosen-future/composite-step' + str(step) + '.png'))

    # Majority vote
    vote_obs_boot = torch.mode(imagined_image, dim=0)[0]
    vote_obs_rpf = torch.mode(imagined_image_rpf, dim=0)[0]
    # Get accuracies
    total_acc_vote_boot.append((vote_obs_boot == target_state).to(torch.float32).mean())
    total_acc_vote_rpf.append((vote_obs_rpf == target_state).to(torch.float32).mean())
    vote_obs_boot = processImgToSave(vote_obs_boot)
    if not os.path.isdir(join('samples-boot/chosen-future')):
        os.makedirs(join('samples-boot/chosen-future'))
    save_image(vote_obs_boot,
               join('samples-boot/chosen-future/vote-step' + str(step) + '.png'))
    vote_obs_rpf = processImgToSave(vote_obs_rpf)
    if not os.path.isdir(join('samples-rpf/chosen-future')):
        os.makedirs(join('samples-rpf/chosen-future'))
    save_image(vote_obs_rpf,
               join('samples-rpf/chosen-future/vote-step' + str(step) + '.png'))

    # Sampled
    sampled_obs_boot = torch.tensor(np.choose(np.random.randint(imagined_image.shape[0], size=imagined_image.shape[1]), imagined_image.cpu().numpy())).to(device)
    sampled_obs_rpf = torch.tensor(np.choose(np.random.randint(imagined_image_rpf.shape[0], size=imagined_image_rpf.shape[1]), imagined_image_rpf.cpu().numpy())).to(device)
    total_acc_sampled_boot.append((sampled_obs_boot == target_state).to(torch.float32).mean())
    total_acc_sampled_rpf.append((sampled_obs_rpf == target_state).to(torch.float32).mean())
    sampled_obs_boot = processImgToSave(sampled_obs_boot)
    if not os.path.isdir(join('samples-boot/chosen-future')):
        os.makedirs(join('samples-boot/chosen-future'))
    save_image(sampled_obs_boot,
               join('samples-boot/chosen-future/sampled-step' + str(step) + '.png'))
    sampled_obs_rpf = processImgToSave(sampled_obs_rpf)
    if not os.path.isdir(join('samples-rpf/chosen-future')):
        os.makedirs(join('samples-rpf/chosen-future'))
    save_image(sampled_obs_rpf,
               join('samples-rpf/chosen-future/sampled-step' + str(step) + '.png'))

print('Total trajectory error (individual head average) Bootstrapped: {} Rpf: {}'.format(np.sum(total_err_head_boot), np.sum(total_err_head_rpf)))
print('Total trajectory error (head avg composite) Bootstrapped: {} Rpf: {}'.format(np.sum(total_err_composite_boot), np.sum(total_err_composite_rpf)))
print('Total trajectory accuracy (majority vote) Bootstrapped: {} Rpf: {}'.format(np.sum(total_acc_vote_boot), np.sum(total_acc_vote_rpf)))
print('Total trajectory accuracy (sampled) Bootstrapped: {} Rpf: {}'.format(np.sum(total_acc_sampled_boot), np.sum(total_acc_sampled_rpf)))