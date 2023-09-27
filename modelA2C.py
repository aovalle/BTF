import torch
import torch.nn as nn
import torch.nn.functional as F

class ACNetwork(nn.Module):
    def __init__(self, representation_shape, num_actions):
        torch.manual_seed(0)
        super(ACNetwork, self).__init__()
        self.input_shape = representation_shape
        channels = representation_shape[0]

        # architecture from higgsfield I2A pacman
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(channels, 16, kernel_size=3, stride=1),
        #     nn.ReLU(),
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(16, 16, kernel_size=3, stride=2),
        #     nn.ReLU(),
        # )
        self.fc = nn.Sequential(
            nn.Linear(self.conv_output_size(), 256),
            nn.ReLU(),
        )

        # Double headed
        self.critic = nn.Linear(256, 1)
        self.actor = nn.Linear(256, num_actions)

        # Init weights
        # self.conv.apply(self.init_weights_kaimingU)
        # self.fc.apply(self.init_weights_kaimingU)
        # self.critic.apply(self.init_weights_kaimingU)
        # self.actor.apply(self.init_weights_kaimingU)

        # Minigrid solver arch
        # self.conv = nn.Sequential(
        #     nn.Conv2d(channels, 16, (2, 2)),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2, 2)),
        #     nn.Conv2d(16, 32, (2, 2)),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, (2, 2)),
        #     nn.ReLU()
        # )
        # self. fc = nn.Sequential(
        #     nn.Linear(self.conv_output_size(), 64),
        #     nn.Tanh(),
        # )
        # # Double headed
        # self.critic = nn.Linear(64, 1)
        # self.actor = nn.Linear(64, num_actions)

    def forward(self, x):
        x = self.conv(x)    # Pass through the conv net
        # PARALLEL
        x = x.view(x.size(0), -1) # Flatten (turn into 1-D per multiprocessing agent)
        # NON-PARALLEL
        #x = x.view(1, -1)   # Flatten (to pass it through the fc), turn into 1-D and take -1 to infer the rest
        #x = x.view(-1)     # Can I just do this?
        x = self.fc(x)      # Go through FC
        probs = F.softmax(self.actor(x), dim=x.dim()-1)     # Get prob for each action
        #logit = self.actor(x)
        value = self.critic(x)
        return probs, value

    ''' Just a function to get the size of the output of the conv net part so it can be specified as input to the FC part'''
    def conv_output_size(self):
        return self.conv(torch.zeros(1,*self.input_shape)).numel()
        #return self.conv(torch.zeros(1, *self.in_shape)).view(1, -1).size(1)


