import math
import random
import warnings

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
import itertools
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

Prospect_Transition = namedtuple('Transition',
                                 ('state', 'action', 'p_next_states','next_states', 'reward'))
class ReplayMemory_Prospect(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Prospect_Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



# DQN ----------------
# from risky_overcooked_rl.utils.equilibrium_solver import NashEquilibriumECOSSolver

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



# DQN ----------------
# from equilibrium_solver import NashEquilibriumECOSSolver
import nashpy as nash
class SelfPlay_NashDQN(object):
    def __init__(self, obs_shape, n_actions,config,**kwargs):
        self.size_hidden_layers = config['size_hidden_layers']
        self.learning_rate = config['lr']
        self.device = config['device']
        self.gamma = config['gamma']
        self.tau = config['tau']

        # super(SelfPlay_NashDQN, self).__init__()
        # self.layer1 = nn.Linear(obs_shape[0], self.size_hidden_layers)
        # self.layer2 = nn.Linear(self.size_hidden_layers, self.size_hidden_layers)
        # self.layer3 = nn.Linear(self.size_hidden_layers, n_actions)
        # # self.mlp_activation = F.relu
        # self.mlp_activation = F.leaky_relu

        self.model = DQN_vector_feature(obs_shape, n_actions,self.size_hidden_layers)

        self.joint_action_space =  list(itertools.product(Action.ALL_ACTIONS, repeat=2))
        self.joint_action_dim = n_actions
        self.player_action_dim = int(np.sqrt(n_actions))
        self.num_agents = 2


        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, amsgrad=True)
        self._transition =  namedtuple('Transition', ('state', 'action', 'reward','next_state','done'))
        self._memory = deque([], maxlen=config['replay_memory_size'])
        self._memory_batch_size = config['minibatch_size']

        self.model = DQN_vector_feature(obs_shape, n_actions, self.size_hidden_layers).to(self.device)
        self.target = DQN_vector_feature(obs_shape, n_actions, self.size_hidden_layers).to(self.device)
        self.target.load_state_dict(self.model.state_dict())


    ###################################################
    ## Memory #########################################
    ###################################################
    def memory_double_push(self, state, action, rewards, next_state, done):
        """ Push both agent's experience into memory from ego perspective"""
        if not isinstance(action, torch.Tensor): action = torch.tensor(action, dtype=torch.int64, device=self.device).reshape(1,1).to(self.device)
        if not isinstance(done, torch.Tensor): done = torch.tensor(done, dtype=torch.int64,device=self.device).reshape(1,1).to(self.device)
        rewards = rewards.flatten()

        # Append Agent 1 experiences
        reward = torch.tensor([rewards[0]], dtype=torch.float32,device=self.device).reshape(1,1).to(self.device)

        # assert len(reward.shape)==2,f'reward shape should be 2D:{reward.shape}'
        self._memory.append(self._transition(state, action, reward, next_state, done))

        # # Append Agent 2 experience
        s_prime = self.invert_obs(state)
        a_prime = self.invert_joint_action(action).to(self.device)
        r_prime = torch.tensor([rewards[1]], dtype=torch.float32,device=self.device).reshape(1,1).to(self.device)
        ns_prime = self.invert_obs(next_state)
        assert len(r_prime.shape) == 2, f'reward shape should be 2D:{r_prime.shape}'
        self._memory.append(self._transition(s_prime, a_prime, r_prime, ns_prime, done))

    def memory_sample(self):
        return random.sample(self._memory, self._memory_batch_size)

    @property
    def memory_len(self):
        return len(self._memory)

    ###################################################
    # Self-Play Utils ######################################
    ###################################################
    def invert_obs(self, obs_batch):
        N_PLAYER_FEAT = 9
        obs_batch = torch.cat([obs_batch[:, N_PLAYER_FEAT:2 * N_PLAYER_FEAT],
                               obs_batch[:, :N_PLAYER_FEAT],
                               obs_batch[:, 2 * N_PLAYER_FEAT:]], dim=1)
        return obs_batch

    def invert_joint_action(self, action_batch):
        BATCH_SIZE = action_batch.shape[0]
        action_batch = torch.tensor(
            [Action.reverse_joint_action_index(action_batch[i]) for i in range(BATCH_SIZE)]).unsqueeze(1)
        return action_batch



    ###################################################
    # Nash Utils ######################################
    ###################################################

    def get_normal_form_game(self,obs):
        """ Batch compute the NF games for each observation"""
        batch_size = obs.shape[0]
        q_values_flat = np.zeros([batch_size,self.num_agents,self.joint_action_dim])
        for i in range(self.num_agents):
            if i==1: obs = self.invert_obs(obs)
            q_values = self.model(obs).detach().cpu().numpy()
            q_values_flat[:,i,:] = q_values
        q_tables = q_values_flat.reshape(batch_size,self.num_agents,
                                         self.player_action_dim,  self.player_action_dim)
        return q_tables

    def solve_nash_eq(self,nf_game, stop_on_first_eq=True):
        """ Solve a single game using Lemke Housen"""
        game = nash.Game(nf_game[0, :], nf_game[1, :])
        np.seterr(all='raise')
        eqs = []
        for label in range(self.player_action_dim + self.player_action_dim):
            try:
                _pi = game.lemke_howson(initial_dropped_label=label)
                if _pi[0].shape == (self.player_action_dim,) and _pi[1].shape == (self.player_action_dim,):
                    if any(np.isnan(_pi[0])) is False and any(np.isnan(_pi[1])) is False:
                        # mixed_eq = _pi; break # find first eqaulib
                        eqs.append(_pi)  # find all equalibs
                        if stop_on_first_eq: break
            except: pass
        np.seterr(all='warn')
        pareto_efficiencies = [np.sum(game[eq[0], eq[1]]) for eq in eqs]
        best_eq_idx = pareto_efficiencies.index(max(pareto_efficiencies))
        dist = np.abs(eqs[best_eq_idx])
        value = game[dist[0], dist[1]]
        return dist, value

    def compute_nash(self, NF_Games, update=False):
        NF_Games = NF_Games.reshape(-1,self.num_agents, self.player_action_dim,  self.player_action_dim)
        all_joint_actions = []
        all_dists = []
        all_ne_values = []

        # Compute nash equilibrium for each game
        for nf_game in NF_Games:
            dist, value = self.solve_nash_eq(nf_game)
            all_dists.append(dist)
            all_ne_values.append(value)

        if update:
            return all_dists, all_ne_values
        else:
            # Sample actions from Nash strategies
            for ne in all_dists:
                # actions = []
                a1 = np.random.choice(np.arange(self.player_action_dim), p=ne[0])
                a2 = np.random.choice(np.arange(self.player_action_dim), p=ne[1])
                action_idxs = (a1, a2)
                joint_action_idx = Action.INDEX_TO_ACTION_INDEX_PAIRS.index(action_idxs)
                all_joint_actions.append(joint_action_idx)
                # for dist in ne:  # iterate over agents
                #     try:
                #         sample_hist = np.random.multinomial(1, dist)  # return one-hot vectors as sample from multinomial
                #     except:
                #         print('Not a valid distribution from Nash equilibrium solution: ', dist)
                #     a = np.where(sample_hist>0)
                #     actions.append(a)
                # all_actions.append(np.array(actions).reshape(-1))

            return np.array(all_joint_actions), all_dists, all_ne_values

    def choose_joint_action(self, obs, epsilon=0.0,debug=False):
        sample = random.random()
        if sample < epsilon: # Explore
            action_probs = np.ones(self.joint_action_dim) / self.joint_action_dim
            joint_action_idx = np.random.choice(np.arange(self.joint_action_dim), p=action_probs)
            joint_action = self.joint_action_space[joint_action_idx]
        else:  # Exploit
            with torch.no_grad():
                NF_Game = self.get_normal_form_game(obs)
                joint_action_idx, dists, ne_vs = self.compute_nash(NF_Game)
                # joint_action_idx = Action.INDEX_TO_ACTION_INDEX_PAIRS.index(action_idxs)
                joint_action = self.joint_action_space[joint_action_idx[0]]
                # if debug: print('Exploit')

        return joint_action,joint_action_idx

    def update(self):
        if self.memory_len < self._memory_batch_size:
            return None

        DoubleTrick = False
        # state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        # state, action, reward, next_state, done = self.memory_sample()
        transitions = self.memory_sample()
        batch = self._transition(*zip(*transitions))
        # BATCH_SIZE = len(transitions)
        state = torch.cat(batch.state)
        next_state = torch.cat(batch.next_state)
        action =  torch.cat(batch.action)
        reward =  torch.cat(batch.reward)
        done =  torch.cat(batch.done)

        # Q-Learning with target network
        # q_values = self.model(state) # .unsqueeze(0)
        q_value = self.model(state).gather(1, action)

        # target_next_q_values_ = self.target(next_state)
        # target_next_q_values = target_next_q_values_.detach().cpu().numpy()

        # target_next_q_values = target_next_q_values_.detach().cpu().numpy()
        # target_next_q_values_ = self.model(next_state) if DoubleTrick else self.target(next_state)
        # target_next_q_values = target_next_q_values_.detach().cpu().numpy()

        # action_ = torch.LongTensor([a[0] * self.action_dim + a[1] for a in action]).to(self.device)
        # q_value = q_values.gather(1, action_.unsqueeze(1)).squeeze(1)

        # solve matrix Nash equilibrium

        try:  # nash computation may encounter error and terminate the process
            NF_games = self.get_normal_form_game(next_state)
            next_dist, next_q_value = self.compute_nash(NF_games, update=True)
            next_q_value = np.array(next_q_value)[:,1].reshape(-1,1)   # only need ego agen q-value
        except:
            warnings.warn("Invalid nash computation.")
            next_q_value = np.zeros_like(reward)

        # if DoubleTrick:  # calculate next_q_value using double DQN trick
        #     next_dist = np.array(next_dist)  # shape: (#batch, #agent, #action)
        #     target_next_q_values = target_next_q_values.reshape((-1, self.action_dim, self.action_dim))
        #     left_multi = np.einsum('na,nab->nb', next_dist[:, 0], target_next_q_values)  # shape: (#batch, #action)
        #     next_q_value = np.einsum('nb,nb->n', left_multi, next_dist[:, 1])

        next_q_value = torch.FloatTensor(next_q_value).to(self.device)
        expected_q_value = reward + (self.gamma) * next_q_value * (1 - done)
        # expected_q_value = reward + (self.gamma ** self.multi_step) * next_q_value * (1 - done)


        # Optimize ----------------
        loss = F.mse_loss(q_value, expected_q_value.detach(), reduction='none')
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        # torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()


        # Perform Soft update on model ----------------
        self.target= soft_update(self.model, self.target, self.tau)
        return loss.item()

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

def optimize_model_prospects(agent_pair,optimizer,transitions,GAMMA,  player=0):
    policy_net = agent_pair.agents[player].policy_net
    target_net = agent_pair.agents[player].target_net

    N_PLAYER_FEAT = 9 # number of features for each player
    batch = Prospect_Transition(*zip(*transitions))
    BATCH_SIZE = len(transitions)

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_states)), device=device, dtype=torch.bool)
    # non_final_next_states = torch.cat([s for s in batch.next_states  if s is not None])
    # non_final_next_state_prospects =  [s for s in batch.next_states if s is not None]
    # non_final_next_state_probs = [s for s in batch.p_next_states if s is not None]
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
        # non_final_next_states = torch.cat([non_final_next_states[:, N_PLAYER_FEAT:2 * N_PLAYER_FEAT],
        #                                    non_final_next_states[:, :N_PLAYER_FEAT],
        #                                    non_final_next_states[:, 2 * N_PLAYER_FEAT:]], dim=1)


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
        for ib, non_final in enumerate(non_final_mask):
            if non_final:
                ns_prospects = batch.next_states[ib]
                p_ns_prospects = batch.p_next_states[ib]
                ns_val = 0
                for j,next_state in enumerate(ns_prospects):
                    if player==1: # invert the player features
                        next_state=torch.cat([next_state[:, N_PLAYER_FEAT:2 * N_PLAYER_FEAT],
                                               next_state[:, :N_PLAYER_FEAT],
                                               next_state[:, 2 * N_PLAYER_FEAT:]], dim=1)

                    p_ns = p_ns_prospects[j]
                    _, next_action_info = agent_pair.action(next_state,use_target_net=True)
                    joint_action_prob = next_action_info['action_probs']
                    joint_action_Q = next_action_info['joint_action_Q'][player]
                    vals_st_prime = np.sum(joint_action_prob * joint_action_Q)
                    ns_val += p_ns * vals_st_prime
                next_state_values[ib] = ns_val

        # agent_pair.action(next_state)
        # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    TD_Target = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, TD_Target.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

