import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
import torch
import os
from risky_overcooked_py.mdp.actions import Action

import yaml
import os
import json


class Config(object):
    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)

    def __repr__(self):
        disp = ''
        for key,val in self.__dict__.items():
            disp += '\n'
            if not type(val) == Config:  disp += '\t'
            disp+= f'{key}: {val}'
        return disp


def dict2obj(dict1):
    # using json.loads method and passing json.dumps
    # method and custom object hook as arguments
    return json.loads(json.dumps(dict1), object_hook=Config)


def get_src_dir():
    dirs = os.getcwd().split('\\')
    src_idx = dirs.index('src') # find index of src directory
    return '\\'.join(dirs[:src_idx+1])


def parse_config():
    src_dir = get_src_dir()
    with open(f'{src_dir}\\risky_overcooked_rl\\algorithms\\MADDPG\\_config.yaml') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    obj=json.loads(json.dumps(config), object_hook=Config)
    return obj

#
# def invert_obs(obs):
#     N_PLAYER_FEAT = 9
#     if isinstance(obs, np.ndarray):
#         _obs = np.concatenate([obs[N_PLAYER_FEAT:2 * N_PLAYER_FEAT],
#                                obs[:N_PLAYER_FEAT],
#                                obs[2 * N_PLAYER_FEAT:]])
#     elif isinstance(obs, torch.Tensor):
#         n_dim = len(obs.shape)
#         if n_dim == 1:
#             _obs = torch.cat([obs[N_PLAYER_FEAT:2 * N_PLAYER_FEAT],
#                                obs[:N_PLAYER_FEAT],
#                                obs[2 * N_PLAYER_FEAT:]])
#         elif n_dim == 2:
#             _obs = torch.cat([obs[:, N_PLAYER_FEAT:2 * N_PLAYER_FEAT],
#                                    obs[:, :N_PLAYER_FEAT],
#                                    obs[:, 2 * N_PLAYER_FEAT:]], dim=1)
#         else: raise ValueError("Invalid obs dimension")
#     return _obs
#
#
#
# def invert_joint_action(joint_action_batch):
#     if isinstance(joint_action_batch, int):
#         return Action.reverse_joint_action_index(joint_action_batch)
#     elif isinstance(joint_action_batch, np.ndarray):
#         BATCH_SIZE = joint_action_batch.shape[0]
#         action_batch = np.array([Action.reverse_joint_action_index(joint_action_batch[i]) for i in range(BATCH_SIZE)])
#     elif isinstance(joint_action_batch, torch.Tensor):
#         BATCH_SIZE = joint_action_batch.shape[0]
#         # action_batch = torch.tensor([Action.reverse_joint_action_index(joint_action_batch[i]) for i in range(BATCH_SIZE)]).unsqueeze(1)
#         action_batch = torch.tensor([Action.reverse_joint_action_index(joint_action_batch[i]) for i in range(BATCH_SIZE)],
#                                     device=joint_action_batch.device).unsqueeze(1)
#     else:  raise ValueError("Invalid joint_action dim")
#     return action_batch
#
#     # BATCH_SIZE = action_batch.shape[0]
#     # action_batch = torch.tensor(
#     #     [Action.reverse_joint_action_index(action_batch[i]) for i in range(BATCH_SIZE)]).unsqueeze(1)
#     # return action_batch

def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
def number_to_onehot(X):
    is_batch = X.dim() == 3
    is_torch = type(X) == torch.Tensor

    if is_batch:
        num_agents = X.shape[1]
        X = X.reshape(-1, 1)

    shape = (X.shape[0], int(X.max() + 1))
    one_hot = np.zeros(shape)
    rows = np.arange(X.shape[0])

    positions = X.reshape(-1).cpu().detach().numpy().astype(int)
    one_hot[rows, positions] = 1.

    if is_batch:
        one_hot = one_hot.reshape(-1, num_agents, int(X.max() + 1))

    if is_torch:
        one_hot = torch.Tensor(one_hot).to(X.device)

    return one_hot
def onehot_to_number(X):
    if type(X) == torch.Tensor:
        return torch.argmax(X, dim=1)
    else:
        return np.argmax(X, axis=1)
class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
    return F.softmax(y / temperature, dim=1)
# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y
