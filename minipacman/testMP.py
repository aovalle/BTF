import numpy as np
from minipacman import MiniPacman
import matplotlib.pyplot as plt

def displayImage(image, step, reward):
    s = "step" + str(step) + " reward " + str(reward)
    plt.title(s)
    plt.imshow(image)
    plt.show()

keys = {'w': 2,'d': 1,'a': 3,'s': 4,'z': 0}

MODES = ('regular', 'avoid', 'hunt', 'ambush', 'rush')
frame_cap = 1000

mode = 'regular'

env = MiniPacman(mode, 1000)

state = env.reset()
# Numpy array - state shape (3,15,19)
done = False

total_reward = 0
step = 1

# Green - pacman, Red - ghost
# State original is: ch X w X h (3,15,19) transformed for displaying into: w X h X c
displayImage(state.transpose(1, 2, 0), step, total_reward)

while not done:
    print("Write input")
    x = input()
    print(x)
    try:
        keys[x]
    except:
        print("Only 'w' 'a' 'd' 's'") # a - left, d - right, w - up, s - down
        continue
    action = keys[x]

    next_state, reward, done, _ = env.step(action)
    print('Inst reward ', reward, done)
    total_reward += reward
    displayImage(next_state.transpose(1, 2, 0), step, total_reward)
    step += 1
