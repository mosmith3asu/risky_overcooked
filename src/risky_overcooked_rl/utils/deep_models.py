import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from risky_overcooked_py.mdp.actions import Action
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

TD_Target_Transition = namedtuple('Transition', ('state', 'action', 'TD_Target'))
class ReplayMemory_CPT(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(TD_Target_Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



# DQN ----------------
class DQN_vector_feature(nn.Module):
    def __init__(self, obs_shape, n_actions,size_hidden_layers,**kwargs):
        self.size_hidden_layers = size_hidden_layers
        super(DQN_vector_feature, self).__init__()
        self.layer1 = nn.Linear(obs_shape[0], self.size_hidden_layers)
        self.layer2 = nn.Linear(self.size_hidden_layers, self.size_hidden_layers)
        self.layer3 = nn.Linear(self.size_hidden_layers, n_actions)
        # self.mlp_activation = F.relu
        self.mlp_activation = F.leaky_relu

        # self.optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.mlp_activation(self.layer1(x))
        x = self.mlp_activation(self.layer2(x))
        x = self.layer3(x) # linear output layer (action-values)
        return x




def soft_update(policy_net, target_net, TAU):
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
    target_net.load_state_dict(target_net_state_dict)
    return target_net

def optimize_model(policy_net,target_net,optimizer,transitions,GAMMA,  player=0):
    N_PLAYER_FEAT = 9 # number of features for each player

    # if len(memory) < BATCH_SIZE:
    #     return
    # transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    BATCH_SIZE = len(transitions)

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state  if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)

    if player==1: # invert the player features
        # switch the first and next 9 player features in state_batch
        state_batch = torch.cat([state_batch[:,N_PLAYER_FEAT:2*N_PLAYER_FEAT],
                                 state_batch[:,:N_PLAYER_FEAT],
                                 state_batch[:,2*N_PLAYER_FEAT:]],dim=1)

        # switch the joint_action index to (partner, ego)
        action_batch = torch.tensor([Action.reverse_joint_action_index(action_batch[i]) for i in range(BATCH_SIZE)]).unsqueeze(1)

        # switch the first and next 9 player features in non_final_next_states
        non_final_next_states = torch.cat([non_final_next_states[:, N_PLAYER_FEAT:2 * N_PLAYER_FEAT],
                                           non_final_next_states[:, :N_PLAYER_FEAT],
                                           non_final_next_states[:, 2 * N_PLAYER_FEAT:]], dim=1)


    reward_batch = torch.cat(batch.reward)
    if len(reward_batch.shape) != 1: # not centralized/solo agent (multi-agent)
        reward_batch = reward_batch[:,player]

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)#.unsqueeze(0)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def optimize_model_td_targets(policy_net,target_net,optimizer,transitions,GAMMA,  player=0):
    N_PLAYER_FEAT = 9 # number of features for each player

    # if len(memory) < BATCH_SIZE:
    #     return
    # transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = TD_Target_Transition(*zip(*transitions))
    BATCH_SIZE = len(transitions)

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    # non_final_next_states = torch.cat([s for s in batch.next_state  if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)


    if player==1: # invert the player features
        # switch the first and next 9 player features in state_batch
        state_batch = torch.cat([state_batch[:,N_PLAYER_FEAT:2*N_PLAYER_FEAT],
                                 state_batch[:,:N_PLAYER_FEAT],
                                 state_batch[:,2*N_PLAYER_FEAT:]],dim=1)

        # switch the joint_action index to (partner, ego)
        action_batch = torch.tensor([Action.reverse_joint_action_index(action_batch[i]) for i in range(BATCH_SIZE)]).unsqueeze(1)

    TD_Target_batch = torch.cat(batch.TD_Target)
    if len(TD_Target_batch.shape) != 1: # not centralized/solo agent (multi-agent)
        TD_Target_batch = TD_Target_batch[:,player]

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)#.unsqueeze(0)

    # # Compute V(s_{t+1}) for all next states.
    # # Expected values of actions for non_final_next_states are computed based
    # # on the "older" target_net; selecting their best reward with max(1).values
    # # This is merged based on the mask, such that we'll have either the expected
    # # state value or 0 in case the state was final.
    # next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # with torch.no_grad():
    #     next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # # Compute the expected Q values
    # expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, TD_Target_batch.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()