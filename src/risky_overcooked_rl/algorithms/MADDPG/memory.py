import numpy as np
import torch
import json, os
from collections import namedtuple, deque
import random
from src.risky_overcooked_rl.algorithms.MADDPG.utils import *
from risky_overcooked_rl.utils.state_utils import invert_obs, invert_joint_action, invert_prospect
# Transition = namedtuple('Transition', ('obs', 'action', 'reward', 'next_obs', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity,device):
        self.memory = deque([], maxlen=capacity)
        self.transition = namedtuple('Transition', ('obs', 'action', 'reward', 'next_obs', 'done'))
        self.device = device

    def push(self, *args):
        """Save a transition"""
        # assert isinstance(args[1],torch.Tensor)
        self.memory.append(self.transition(*args))

    def double_push(self, obs, joint_action_idx, rewards, next_obs, done):
        """ Push both agent's experience into memory from ego perspective"""
        n_agents = 2
        rewards = rewards.flatten()
        joint_action_idx = torch.tensor(joint_action_idx, dtype=torch.int64, device=self.device).reshape(1, 1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).reshape(2, 1).to(self.device)
        done = torch.tensor(done, dtype=torch.int64, device=self.device).reshape(1, 1).to(self.device)

        for i in range(n_agents):
            if i==1: # invert the perspective
                joint_action_idx = invert_joint_action(joint_action_idx)
                obs = invert_obs(obs)
                next_obs = invert_obs(next_obs)
            self.push(obs, joint_action_idx, rewards[i], next_obs, done)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ReplayMemory_Prospect(object):

    def __init__(self, capacity, device):
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_prospects', 'done'))
        self.memory = deque([], maxlen=capacity)
        self.device = device

    def double_push(self, state, action, rewards, next_prospects, done):
        """ Push both agent's experience into memory from ego perspective"""
        if not isinstance(action, torch.Tensor): action = torch.tensor(action, dtype=torch.int64, device=self.device).reshape(1, 1).to(self.device)
        if not isinstance(done, torch.Tensor): done = torch.tensor(done, dtype=torch.int64, device=self.device).reshape(1, 1).to(self.device)
        rewards = rewards.flatten()

        # Append Agent 1 experiences
        reward = torch.tensor([rewards[0]], dtype=torch.float32, device=self.device).reshape(1, 1).to(self.device)

        # assert len(reward.shape)==2,f'reward shape should be 2D:{reward.shape}'
        self.memory.append(self.transition(state, action, reward, next_prospects, done))

        # # Append Agent 2 experience
        s_prime = invert_obs(state)
        a_prime = invert_joint_action(action).to(self.device)
        r_prime = torch.tensor([rewards[1]], dtype=torch.float32, device=self.device).reshape(1, 1).to(self.device)
        np_prime = invert_prospect(next_prospects)
        # assert len(r_prime.shape) == 2, f'reward shape should be 2D:{r_prime.shape}'
        self.memory.append(self.transition(s_prime, a_prime, r_prime, np_prime, done))

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# class ReplayBuffer(object):
#
#     def __init__(self, obs_shape, action_shape, reward_shape, dones_shape, capacity, device):
#         self.capacity = capacity
#         self.device = device
#
#         self.obses = np.empty((capacity, *obs_shape), dtype=np.float32)
#         self.next_obses = np.empty((capacity, *obs_shape), dtype=np.float32)
#         self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
#         self.rewards = np.empty((capacity, *reward_shape), dtype=np.float32)
#         self.dones = np.empty((capacity, *dones_shape), dtype=np.float32)
#
#         self.idx = 0
#         self.full = False
#
#     def __len__(self):
#         return self.capacity if self.full else self.idx
#
#     def add(self, obs, action, reward, next_obs, dones):
#         np.copyto(self.obses[self.idx], obs)
#         np.copyto(self.actions[self.idx], action)
#         np.copyto(self.rewards[self.idx], reward)
#         np.copyto(self.next_obses[self.idx], next_obs)
#         np.copyto(self.dones[self.idx], dones)
#
#         self.idx = (self.idx + 1) % self.capacity
#         self.full = self.full or self.idx == 0
#
#     def sample(self, batch_size, nth=None):
#         idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)
#
#         if nth:
#             obses = torch.FloatTensor(self.obses[idxs][:, nth]).to(self.device)
#             actions = torch.FloatTensor(self.actions[idxs][:, nth]).to(self.device)
#             rewards = torch.FloatTensor(self.rewards[idxs][:, nth]).to(self.device)
#             next_obses = torch.FloatTensor(self.next_obses[idxs][:, nth]).to(self.device)
#             dones = torch.FloatTensor(self.dones[idxs][:, nth]).to(self.device)
#         else:
#             obses = torch.FloatTensor(self.obses[idxs]).to(self.device)
#             actions = torch.FloatTensor(self.actions[idxs]).to(self.device)
#             rewards = torch.FloatTensor(self.rewards[idxs]).to(self.device)
#             next_obses = torch.FloatTensor(self.next_obses[idxs]).to(self.device)
#             dones = torch.FloatTensor(self.dones[idxs]).to(self.device)
#
#         return obses, actions, rewards, next_obses, dones
#
#     def save(self, root_dir, step) -> None:
#         make_dir(root_dir, 'buffer') if root_dir else None
#         length = self.capacity if self.full else self.idx
#         path = os.path.join(root_dir, 'buffer')
#
#         make_dir(path, str(step))
#         path = os.path.join(path, str(step))
#
#         np.savez_compressed(os.path.join(path, 'state.npz'), self.obses)
#         np.savez_compressed(os.path.join(path, 'next_state.npz'), self.next_obses)
#         np.savez_compressed(os.path.join(path, 'action.npz'), self.actions)
#         np.savez_compressed(os.path.join(path, 'reward.npz'), self.rewards)
#         np.savez_compressed(os.path.join(path, 'done.npz'), self.dones)
#
#         info = dict()
#         info['idx'] = self.idx
#         info['capacity'] = self.capacity
#         info['step'] = step
#         info['size'] = length
#
#         with open(os.path.join(path, 'info.txt'), 'w') as f:
#             output = json.dumps(info, indent=4, sort_keys=True)
#             f.write(output)
#
#     def load(self, root_dir) -> None:
#         path = os.path.join(root_dir, 'buffer')
#
#         self.obses = np.load(os.path.join(path, 'state.npz'))['arr_0']
#         self.next_obses = np.load(os.path.join(path, 'next_state.npz'))['arr_0']
#         self.actions = np.load(os.path.join(path, 'action.npz'))['arr_0']
#         self.rewards = np.load(os.path.join(path, 'reward.npz'))['arr_0']
#         self.dones = np.load(os.path.join(path, 'done.npz'))['arr_0']
#
#         with open(os.path.join(path, 'info.txt'), 'r') as f:
#             info = json.load(f)
#
#         self.idx = int(info['idx'])
#         self.capacity = int(info['capacity'])
#         self.full = int(info['step']) >= self.capacity
#
#     def append_data(self, dir_path):
#
#         def loader(path):
#             logger.info('Loading data - ' + path)
#             data =  np.load(path)['arr_0']
#             logger.info('Loaded data - ' + path)
#             return data
#
#         obses_data = loader(os.path.join(dir_path, 'state.npz'))
#         self.obses = np.concatenate((self.obses, obses_data), axis=0)
#
#         next_obses_data = loader(os.path.join(dir_path, 'next_state.npz'))
#         self.next_obses = np.concatenate((self.next_obses, next_obses_data), axis=0)
#
#         reward_data = loader(os.path.join(dir_path, 'reward.npz'))
#         self.rewards = np.concatenate((self.rewards, reward_data), axis=0)
#
#         action_data = loader(os.path.join(dir_path, 'action.npz'))
#         self.actions = np.concatenate((self.actions, action_data), axis=0)
#
#         done_data = loader(os.path.join(dir_path, 'done.npz'))
#         self.dones = np.concatenate((self.dones, done_data), axis=0)
#
#         if self.idx == 0:
#             self.idx = -1
#         self.idx += len(obses_data)