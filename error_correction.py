import numpy as np
import torch

from params import Params as p
from utils.misc import prepAndSaveImg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

elems = {0:'fruit', 1:'pacman', 6:'ghost', 5:'eaten cell'}

'''
pred_image:         unified prediction computed from the ensemble
heads_pred_image:   prediction per head
'''
def correct_missing(pred_image, heads_pred_image, element, var_name, curr_pos, retpos=False):
    # Get positions where element is located in those heads where it's present
    positions = (heads_pred_image == element).nonzero()
    if 'sample' in var_name:
        # Sample a position
        try:
            pos = np.random.choice(positions[:, 1].cpu())
        # if the element is nowhere to be found get it from last real observed frame
        except:
            pos = curr_pos
    elif 'vote' or 'compo' in var_name:
        try:
            pos = torch.mode(positions[:, 1])[0].item()
        except:
            pos = curr_pos

    prepAndSaveImg(pred_image, '.', elems[element] + '-mis-preEC.png')
    # Place the element
    pred_image[pos] = element
    prepAndSaveImg(pred_image, '.', elems[element] + '-mis-postEC.png')
    if not retpos:
        return pred_image
    else:
        return pred_image, pos


def correct_additional(pred_image, heads_pred_image, element, var_name, past_real_image, retpos=False):
    # Get where duplicated elements are located in the unified ensemble prediction
    pred_positions = (pred_image == element).nonzero()
    # Get predicted ghosts locations from heads
    inheads_positions = (heads_pred_image == element).nonzero()

    # Check which of the duplicated elements from the final prediction, where are they in the
    # predictions of the heads. This is done to limit ourselves only to those values from the heads
    # that made it to the final unified prediction
    idx = np.isin(inheads_positions[:, 1].cpu(), pred_positions.cpu())
    if 'sample' in var_name:
        # Sample a position from heads to determine what element should stay
        pos = np.random.choice(inheads_positions[idx, 1].cpu())
    elif 'vote' or 'compo' in var_name:
        pos = torch.mode(inheads_positions[idx, 1])[0].item()

    # TODO: ADD SPECIAL IDENTIFIER
    prepAndSaveImg(pred_image, '.', elems[element]+'-add-preEC.png')
    prepAndSaveImg(past_real_image, '.', elems[element]+'-add-past.png')

    # The position of the elements that won't be kept
    nonchosen = pred_positions[pred_positions != pos]

    # TODO: CHECK IF THIS PREVIOUS STEP IS GOOD OR IF CHECKING FOR DUPLICATES
    #  AND USING ENSEMBLE PREDICTIONS ONLY IS BETTER
    if element == p.GHOST:
        # After taking the position of a ghost to be kept, we take the positions of those that won't
        # and set them to the value of the last observed real frame
        #pred_image[pred_positions[pred_positions != pos]] = past_real_image[pred_positions[pred_positions != pos]]
        pred_image[nonchosen] = past_real_image[nonchosen]
        # try:
        #     pred_image[nonchosen] = past_real_image[nonchosen]
        # except:
        #     print(pred_image[nonchosen], past_real_image[nonchosen])
        #     print(pred_image, past_real_image)
        #     print('exception')
        #     pred_image[nonchosen] = past_real_image[nonchosen]
        #     exit()
        #     pred_image[nonchosen] = past_real_image[nonchosen].to(device)
    elif element == p.PACMAN:
        # From the last real frame, extract what elements were in those positions where we imagined
        # there should be a pacman
        idx_pac = (past_real_image[nonchosen] == p.PACMAN)
        idx_nopac = (past_real_image[nonchosen] != p.PACMAN)
        # split them between those positions that had a pacman and those that didnt...
        pos_to_eaten = nonchosen[idx_pac]
        pos_to_prev = nonchosen[idx_nopac]
        # ... because if there was a pacman in that position then now it should turn black
        # and if there wasn't then just set it to its previous value
        pred_image[pos_to_eaten] = p.EATEN_CELL
        pred_image[pos_to_prev] = past_real_image[pos_to_prev]
        # try:
        #     pred_image[pos_to_prev] = past_real_image[pos_to_prev]
        # except:
        #     print(pred_image, past_real_image)
        #     print(pred_image[pos_to_prev], past_real_image[pos_to_prev])
        #     exit()


        # try:
        #     pred_image[pos_to_prev] = past_real_image[nonchosen]
        # except:
        #     if pred_image[pos_to_prev].shape[0] == 0:
        #         pass
        #     else:
        #         #TODO: CHECK THE EXCEPTION THIS DEFINITELY LOOKS WRONG!!!!!!!
        #         #TODO HIGHEST PRIORITY
        #         print(pos_to_prev, pos_to_eaten, nonchosen, idx_pac, idx_nopac,
        #               pred_image[pos_to_eaten], pred_image[pos_to_prev])
        #         print('check dimensions ', pred_image[pos_to_prev].shape[0])
        #         pred_image[pos_to_prev] = past_real_image[nonchosen]
        #         # FIXME: this should be it but check:
        #         #pred_image[pos_to_prev] = past_real_image[pos_to_prev]

    prepAndSaveImg(pred_image, '.', elems[element]+'-add-postEC.png')

    # Re-check again that in those positions where the elements should have been removed the element
    # is really no longer present
    duplicated = (pred_image[nonchosen] == element).nonzero()
    # But if they're, check what position in the unified prediction frame would that correspond to
    leftover_pos = nonchosen[duplicated]
    # Now check if in fact there are still elements to be corrected
    # In that case we're goint to extract information from the heads
    if leftover_pos.nelement() > 0:
        if element == p.GHOST:
            pred_image = correct_ghosts(leftover_pos, pred_image, heads_pred_image, element, var_name)
        elif element == p.PACMAN:
            print(pos, nonchosen, idx_pac, idx_nopac, pos_to_eaten, pos_to_prev)
            correct_pacmans()

    if not retpos:
        return pred_image   # return corrected predicted frame
    else:
        return pred_image, pos

'''
Corrects ghosts with information from the prediction of the heads
'''
def correct_ghosts(leftover_pos, pred_image, heads_pred_image, element, var_name):
    prepAndSaveImg(pred_image, '.', elems[element]+'-dup-preEC.png')
    for pos in leftover_pos:
        try:
            if 'sampled' in var_name:
                newValue = np.random.choice(heads_pred_image[:, pos][heads_pred_image[:, pos] != element].cpu()).item()
            elif 'vote' or 'compo' in var_name:
                newValue = torch.mode(heads_pred_image[:, pos][heads_pred_image[:, pos] != element])[0].item()
        # Exception if tensors above were empty (e.g. no information that could be derived from the heads
        # because they were actually full of ghosts/pacman so we can't get an alternative value)
        except:
            newValue = element

        pred_image[pos] = newValue

    prepAndSaveImg(pred_image, '.', elems[element]+'-dup-postEC.png')
    return pred_image

def correct_pacmans():
    raise Exception('more than one pacman. Not implemented!!!!!!')