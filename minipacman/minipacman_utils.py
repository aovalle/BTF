import numpy as np

#7 different pixels in MiniPacman
pixels = (
    (0.0, 1.0, 1.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (1.0, 1.0, 0.0),
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
)

# Convert each pixel to a category (e.g: pix:1, pix:2...)
pixel_to_categorical = {pix:i for i, pix in enumerate(pixels)}
num_pixels = len(pixels)

#For each mode in MiniPacman there are different rewards
mode_rewards = {
    "regular": [-1, 0, 1, 3, 5, 6, 8],
    #"regular": [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "avoid":   [0.1, -0.1, -5, -10, -20],
    "hunt":    [0, 1, 10, -20],
    "ambush":  [0, -0.1, 10, -20],
    "rush":    [0, -0.1, 9.9]
}
# Convert to categorical (e.g.: mode:{rew:0, rew1:1, rew2:2...}
reward_to_categorical = {mode: {reward:i for i, reward in enumerate(mode_rewards[mode])} for mode in mode_rewards.keys()}
num_rewards = len(mode_rewards["regular"])


''' transforms state frames into a target shape 
first it transforms the 4-D array (batch, channel, w, h) to a 2-D array ((batch*w*h), channel)
thus rows=# of pixels in frames (times number of batches), cols=# of values that a pixel can take (one col per channel)
then it transforms those 3 col of the channel to a single value since it transforms it to categorical
the end result is a list of 285 ints (time number of batches) with the value corresponding to its category as per above'''
def pix_to_target(states):
    target = []
    for pixel in states.transpose(0, 2, 3, 1).reshape(-1, 3):
        target.append(pixel_to_categorical[tuple([np.ceil(pixel[0]), np.ceil(pixel[1]), np.ceil(pixel[2])])])
    return target


def target_to_pix(imagined_states):
    pixels = []
    to_pixel = {value: key for key, value in pixel_to_categorical.items()}
    for target in imagined_states:
        pixels.append(list(to_pixel[target]))
    return np.array(pixels)

''' converts a reward into its categorical form (see mode rewards above)'''
def rewards_to_target(mode, rewards):
    target = []
    # If we're only receiving one int instead of an array (e.g. not parallelizing, no batch)
    if isinstance(rewards, int):
        target.append(reward_to_categorical[mode][rewards])
    else:
        for reward in rewards:
            target.append(reward_to_categorical[mode][reward])
    return target