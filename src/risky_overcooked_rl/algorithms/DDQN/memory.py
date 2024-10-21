import random
from collections import namedtuple, deque
import torch
# from risky_overcooked_rl.algorithms.DDQN.utils import invert_obs, invert_joint_action, invert_prospect
from risky_overcooked_rl.utils.state_utils import invert_obs, invert_joint_action, invert_prospect
# REPLAY MEMORY ----------------
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# TD_Target_Transition = namedtuple('Transition', ('state', 'action', 'TD_Target'))
# class ReplayMemory_CPT(object):
#
#     def __init__(self, capacity):
#         self.memory = deque([], maxlen=capacity)
#
#     def push(self, *args):
#         """Save a transition"""
#         self.memory.append(TD_Target_Transition(*args))
#
#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)
#
#     def __len__(self):
#         return len(self.memory)


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
