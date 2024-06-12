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


def select_action(state,policy_net,exp_prob,n_actions=6,debug=False,rationality=None):
    # global steps_done
    sample = random.random()
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #     math.exp(-1. * steps_done / EPS_DECAY)
    # steps_done += 1
    if rationality is not None:
        if rationality == 'max':
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # return policy_net(state).max(1).indices.view(1, 1).numpy().flatten()[0]
                return policy_net(state).max(1).indices.view(1, 1)
        else:
            with torch.no_grad():
                qA = policy_net(state).numpy().flatten()
                ex = np.exp(rationality*(qA-np.max(qA)))
                pA = ex/np.sum(ex)
                action = Action.sample(pA)
                ai = Action.ACTION_TO_INDEX[action]
                return torch.tensor([[ai]], device=device, dtype=torch.long)
    elif sample < exp_prob:
        # return np.random.choice(np.arange(n_actions))
        action = Action.sample(np.ones(n_actions)/n_actions)
        ai = Action.ACTION_TO_INDEX[action]
        return torch.tensor([[ai]], device=device, dtype=torch.long)
        # return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    else:
        if debug: print('Greedy')
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # return policy_net(state).max(1).indices.view(1, 1).numpy().flatten()[0]
            return policy_net(state).max(1).indices.view(1, 1)


def soft_update(policy_net, target_net, TAU):
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
    target_net.load_state_dict(target_net_state_dict)
    return target_net
def optimize_model(policy_net,target_net,optimizer,memory,BATCH_SIZE,GAMMA):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

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

# DQN ----------------
class DQN_mask_feature(nn.Module):
    def __init__(self, obs_shape, n_actions,num_filters,size_hidden_layers,**kwargs):
        super(DQN_mask_feature, self).__init__()

        # Parse Config
        # size_hidden_layers = kwargs['size_hidden_layers']
        # num_filters = kwargs['num_filters']
        num_channels,width, height = obs_shape
        num_filters = num_channels ############################ TESTING: #num_filters * width * height
        flatten_size = 34

        # self.model = nn.Sequential(
        #     nn.Conv2d(in_channels=num_channels, out_channels=num_filters, kernel_size=(5, 5), padding='same'), nn.LeakyReLU(),
        #     nn.Conv2d(in_channels=num_channels, out_channels=num_filters, kernel_size=(3, 3), padding='same'), nn.LeakyReLU(),
        #     nn.Conv2d(in_channels=num_channels, out_channels=num_filters, kernel_size=(3, 3), padding='valid'), nn.LeakyReLU(),
        #     nn.Flatten(),
        #     nn.Linear(flatten_size, size_hidden_layers),        nn.LeakyReLU(),
        #     nn.Linear(size_hidden_layers, size_hidden_layers),  nn.LeakyReLU(),
        #     nn.Linear(size_hidden_layers, n_actions),
        # )

        # Model Params
        # self.mlp_activation = F.relu
        self.activation = F.leaky_relu
        self.loss_fn = F.mse_loss
        # self.loss_fn = F.kl_div

        # Define "Vision" CNN layers
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_filters, kernel_size=(5, 5),padding='same')
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_filters, kernel_size=(3, 3),padding='same')
        self.conv3 = nn.Conv2d(in_channels=num_channels, out_channels=num_filters, kernel_size=(3, 3),padding='valid')
        self.flatten4 = nn.Flatten()
        # Define MLP layers

        self.layer1 = nn.Linear(flatten_size, size_hidden_layers)
        self.layer2 = nn.Linear(size_hidden_layers, size_hidden_layers)
        self.layer3 = nn.Linear(size_hidden_layers, n_actions)


        # self.optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.flatten4(x)
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)  # linear output layer (action-values)
        return x

if __name__ == "__main__":
    print(f'Cuda available: {torch.cuda.is_available()}')