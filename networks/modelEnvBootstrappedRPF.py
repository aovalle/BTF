import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicBlock(nn.Module):
    def __init__(self, in_shape, n1, n2, n3):
        super(BasicBlock, self).__init__()
        # Channels x w x h (channels are not state channels though they come from the output channels of a previous convnet)
        self.in_shape = in_shape
        # Hyperparameters determining number of channels in the convs
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.maxpool = nn.MaxPool2d(kernel_size=in_shape[1:])
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_shape[0] * 2, n1, kernel_size=1, stride=2, padding=6),
            nn.ReLU(),
            nn.Conv2d(n1, n1, kernel_size=10, stride=1, padding=(5, 6)),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_shape[0] * 2, n2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(n2, n2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(n1 + n2, n3, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, inputs):
        x = self.pool_and_inject(inputs)                    # Pool and inject
        x = torch.cat([self.conv1(x), self.conv2(x)], 1)    # Pass it through two headed conv nets and concat
        x = self.conv3(x)                                   # Pass it through another conv net
        x = torch.cat([x, inputs], 1)                       # Concat output with original input
        return x

    ''' Pool and inject: size preserving layer that communicates the max value of each layer globally to next convnet'''
    def pool_and_inject(self, x):
        pooled = self.maxpool(x)
        tiled = pooled.expand((x.size(0),) + self.in_shape)
        out = torch.cat([tiled, x], 1)  # Concatenate max-pooled stuff with original input
        return out


class EnvModelBootstrappedNetwork(nn.Module):
    ''' Receives the input space shape, the number of type of pixels, and the number of possible rewards '''
    def __init__(self, in_shape, num_pixels, num_rewards, num_heads):
        super(EnvModelBootstrappedNetwork, self).__init__()
        # Input space = Ch x w x h
        width = in_shape[1]
        height = in_shape[2]

        self.conv = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=1),
            nn.ReLU()
        )

        self.basic_block1 = BasicBlock((64, width, height), 16, 32, 64)
        self.basic_block2 = BasicBlock((128, width, height), 16, 32, 64)

        # Image head
        self.bootstrapped_obs_conv = []
        self.bootstrapped_obs_fc = []
        for head in range(num_heads):
            obs_conv = nn.Sequential(
                nn.Conv2d(192, 256, kernel_size=1),     # input from the basic blocks is 192
                nn.ReLU()
            )
            obs_fc = nn.Linear(256, num_pixels)
            # Lists of head archs
            self.bootstrapped_obs_conv.append(obs_conv)
            self.bootstrapped_obs_fc.append(obs_fc)
            self.__setattr__("head_{}_obs_conv".format(head), obs_conv)
            self.__setattr__("head_{}_obs_fc".format(head), obs_fc)


        # Reward heads
        self.bootstrapped_reward_conv = []
        self.bootstrapped_reward_fc = []
        for head in range(num_heads):
            reward_conv = nn.Sequential(
                nn.Conv2d(192, 64, kernel_size=1),      # input from the basic blocks is 192
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=1),
                nn.ReLU()
            )
            reward_fc = nn.Linear(64 * width * height, num_rewards)
            # Lists of head archs
            self.bootstrapped_reward_conv.append(reward_conv)
            self.bootstrapped_reward_fc.append(reward_fc)
            # NOTE: this was actually crucial (AFAIK) to construct the graph properly, otherwise it won't work
            self.__setattr__("head_{}_reward_conv".format(head), reward_conv)
            self.__setattr__("head_{}_reward_fc".format(head), reward_fc)

    def forward(self, inputs, head_idxs):
        batch_size = inputs.size(0)

        x = self.conv(inputs)
        x = self.basic_block1(x)
        x = self.basic_block2(x)

        # image = self.image_conv(x)
        # image = image.permute(0, 2, 3, 1).contiguous().view(-1, 256)
        # image = self.image_fc(image)    # output is a row for each pixel (15x19=285) and cols are # of type of pixels (7)
        # reward = self.reward_conv(x)
        # reward = reward.view(batch_size, -1)
        # reward = self.reward_fc(reward)  # output is a single row with cols are # of possible rewards in this mode

        rewards = []
        if type(head_idxs) is not list:
            head_idxs = [head_idxs]
        for idx in head_idxs:
            reward = self.bootstrapped_reward_conv[idx](x)
            reward = reward.view(batch_size, -1)
            reward = self.bootstrapped_reward_fc[idx](reward) # output is a single row with cols are # of possible rewards in this mode
            rewards.append(reward.detach())  # append head output #TODO FIND A WAY TO DO THIS MORE ELEGANTLY

        obsvs = []
        if type(head_idxs) is not list:
            head_idxs = [head_idxs]
        for idx in head_idxs:
            obs = self.bootstrapped_obs_conv[idx](x)
            obs = obs.permute(0, 2, 3, 1).contiguous().view(-1, 256)
            obs = self.bootstrapped_obs_fc[idx](obs) # output is a row for each pixel (15x19=285) and cols are # of type of pixels (7)
            obsvs.append(obs.detach())

        return obsvs, rewards


''' Randomized Prior Functions '''
class EnvModelRPF(nn.Module):
    def __init__(self, base_model, prior_model, prior_scale=1.0):
        super().__init__()
        self.base_model = base_model
        self.prior_model = prior_model
        self.prior_scale = prior_scale

    def forward(self, inputs, head_idxs):
        # RPF Model
        with torch.no_grad():
            prior_obs, prior_rewards = self.prior_model(inputs, head_idxs)#, rpf=True) #TODO:if i set it to nograd then do i still need the detach()?
            #prior_obs = prior_obs.detach()
            #prior_rewards = prior_rewards.detach()

        # Bootstrapped model
        obsvs, rewards = self.base_model(inputs, head_idxs)

        # print(rewards)
        # print(prior_rewards)
        # exit()

        # Combine them with RPF model
        obsvs = np.array(obsvs) + (np.array(self.prior_scale) * prior_obs)
        rewards = np.array(rewards) + (np.array(self.prior_scale) * prior_rewards)

        return obsvs.tolist(), rewards.tolist()