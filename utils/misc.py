import numpy as np
import torch
import os
from os.path import join
from torchvision.utils import save_image

import minipacman.minipacman_utils as mpu


# def saveImg(state, dir, fname):
#     #TODO: FIX THIS
#
#     # for model
#     state_image = state
#
#     # for non model
#     #state_image = torch.FloatTensor(state).unsqueeze(0)  # .permute(1, 2, 0).cpu().numpy()
#
#     #print(state_image.shape)
#     if not os.path.isdir(dir):
#         os.makedirs(dir)
#     save_image(state_image, join(dir + fname))


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