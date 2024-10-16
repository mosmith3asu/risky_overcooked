import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
import torch
import os
from risky_overcooked_py.mdp.actions import Action

def invert_obs(obs):
    N_PLAYER_FEAT = 9
    if isinstance(obs, np.ndarray):
        _obs = np.concatenate([obs[N_PLAYER_FEAT:2 * N_PLAYER_FEAT],
                               obs[:N_PLAYER_FEAT],
                               obs[2 * N_PLAYER_FEAT:]])
    elif isinstance(obs, torch.Tensor):
        n_dim = len(obs.shape)
        if n_dim == 1:
            _obs = torch.cat([obs[N_PLAYER_FEAT:2 * N_PLAYER_FEAT],
                               obs[:N_PLAYER_FEAT],
                               obs[2 * N_PLAYER_FEAT:]])
        elif n_dim == 2:
            _obs = torch.cat([obs[:, N_PLAYER_FEAT:2 * N_PLAYER_FEAT],
                                   obs[:, :N_PLAYER_FEAT],
                                   obs[:, 2 * N_PLAYER_FEAT:]], dim=1)
        else: raise ValueError("Invalid obs dimension")
    return _obs



def invert_joint_action(joint_action_batch):
    if isinstance(joint_action_batch, int):
        return Action.reverse_joint_action_index(joint_action_batch)
    elif isinstance(joint_action_batch, np.ndarray):
        BATCH_SIZE = joint_action_batch.shape[0]
        action_batch = np.array([Action.reverse_joint_action_index(joint_action_batch[i]) for i in range(BATCH_SIZE)])
    elif isinstance(joint_action_batch, torch.Tensor):
        BATCH_SIZE = joint_action_batch.shape[0]
        # action_batch = torch.tensor([Action.reverse_joint_action_index(joint_action_batch[i]) for i in range(BATCH_SIZE)]).unsqueeze(1)
        action_batch = torch.tensor([Action.reverse_joint_action_index(joint_action_batch[i]) for i in range(BATCH_SIZE)],
                                    device=joint_action_batch.device).unsqueeze(1)
    else:  raise ValueError("Invalid joint_action dim")
    return action_batch

    # BATCH_SIZE = action_batch.shape[0]
    # action_batch = torch.tensor(
    #     [Action.reverse_joint_action_index(action_batch[i]) for i in range(BATCH_SIZE)]).unsqueeze(1)
    # return action_batch
