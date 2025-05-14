import random
import torch.optim.lr_scheduler as lr_scheduler
from numba import njit,prange
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from risky_overcooked_py.mdp.actions import Action
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import itertools
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
from risky_overcooked_rl.algorithms.DDQN.utils.memory import ReplayMemory_Prospect
# from risky_overcooked_rl.utils.risk_sensitivity import CumulativeProspectTheory
from risky_overcooked_rl.utils.risk_sensitivity_compiled import CumulativeProspectTheory
from risky_overcooked_rl.utils.model_manager import get_absolute_save_dir
from risky_overcooked_rl.utils.state_utils import invert_obs, flatten_next_prospects
import numpy as np
import warnings
import copy
import os
plt.ion()

# if GPU is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SelfPlay_QRE_OSA(object):
    @classmethod
    def from_file(cls,obs_shape, n_actions, agents_config, fname):

        # instantiate base class -------------
        agents = cls(obs_shape, n_actions, agents_config)

        # find saved models absolute dir -------------
        dir = get_absolute_save_dir()

        # select file to load ---------------
        files = os.listdir(dir)
        files = [f for f in files if (fname in f and '.pt' in f)]
        if len(files) == 0: raise FileNotFoundError(f'No files found with fname:'+fname)
        elif len(files) == 1: loads_fname = files[0]
        elif len(files) > 1:
            warnings.warn(f'Multiple files found with fname: {fname}. Using latest file...')
            loads_fname = files[-1]
        else: raise ValueError('Unexpected error occurred')
        PATH = dir + loads_fname

        print(f'\n#########################################')
        print(f'Loading model from: {loads_fname}')
        print(f'#########################################\n')

        # Load file and update base class ---------
        loaded_model = torch.load(PATH, weights_only=True, map_location=agents_config['model']['device'])
        agents.model.load_state_dict(loaded_model)
        agents.target.load_state_dict(loaded_model)
        agents.checkpoint_model.load_state_dict(loaded_model)
        # is_same = np.all([torch.all(agents.model.state_dict()[key] == agents.model.state_dict()[key]) for key in
        #         agents.model.state_dict().keys()])
        return agents

    def __init__(self, obs_shape, n_actions, agents_config,**kwargs):

        # Instatiate Base Config -------------
        self.rationality = agents_config['rationality']
        self.num_agents = 2
        self.joint_action_space = list(itertools.product(Action.ALL_ACTIONS, repeat=2))
        self.joint_action_dim = n_actions
        self.player_action_dim = int(np.sqrt(n_actions))

        # Parse Equilibrium Config -------------
        eq_config = agents_config['equilibrium']
        self.eq_sol = eq_config['type']

        # Parse NN Model config ---------------
        model_config = agents_config['model']
        self.clip_grad = model_config['clip_grad']
        self.num_hidden_layers = model_config['num_hidden_layers']
        self.size_hidden_layers = model_config['size_hidden_layers']
        self.learning_rate = model_config['lr']
        self.device = model_config['device']
        self.gamma = model_config['gamma']
        self.tau = model_config['tau']
        self.mem_size = model_config['replay_memory_size']
        self.minibatch_size = model_config['minibatch_size']

        # Define Memory
        self._memory = ReplayMemory_Prospect(self.mem_size,self.device)
        self._memory_batch_size = self.minibatch_size

        # Define Model
        self.model = DQN_vector_feature(obs_shape, n_actions,self.num_hidden_layers, self.size_hidden_layers).to(self.device)
        self.target = DQN_vector_feature(obs_shape, n_actions,self.num_hidden_layers, self.size_hidden_layers).to(self.device)
        self.checkpoint_model = DQN_vector_feature(obs_shape, n_actions,self.num_hidden_layers, self.size_hidden_layers).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.checkpoint_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, amsgrad=True)

        # lr_warmup_iter = config['lr_sched'][2]
        # lr_factor = config['lr_sched'][0]/config['lr_sched'][1]
        # lr_factor = self.lr_warmup_scale
        # self.optimizer = optim.AdamW(self.model.parameters(), lr=lr_factor * self.learning_rate, amsgrad=True)
        # self.scheduler = lr_scheduler.LinearLR(self.optimizer,
        #                                        start_factor=1,
        #                                        end_factor=1 / lr_factor,
        #                                        total_iters=lr_warmup_iter)
        # self.optimistic_value_expectation = False
        # if self.optimistic_value_expectation: warnings.warn("Optimistic value expectation is set to True.")

    def update_checkpoint(self):
        self.checkpoint_model.load_state_dict(self.model.state_dict())


    ###################################################
    # Nash Utils ######################################
    ###################################################

    def get_normal_form_game(self, obs, with_model=None):
        """ Batch compute the NF games for each observation"""
        batch_size = obs.shape[0]
        all_games = torch.zeros([batch_size, self.num_agents, self.player_action_dim, self.player_action_dim], device=self.device)
        for i in range(self.num_agents):
            if i == 1: obs = invert_obs(obs)
            if with_model is not None: q_values = with_model(obs).detach()
            else: q_values = self.model(obs).detach()
            # if use_target: q_values = self.target(obs).detach()
            # else: q_values = self.model(obs).detach()
            q_values = q_values.reshape(batch_size, self.player_action_dim, self.player_action_dim)
            all_games[:, i, :, :] = q_values if i == 0 else torch.transpose(q_values, -1, -2)
        return all_games

    def get_expected_equilibrium_value(self, nf_games, dists):
        ego, partner = 0, 1
        joint_dist_mat = torch.bmm(dists[:,ego].unsqueeze(-1), torch.transpose(dists[:,partner].unsqueeze(-1), -1, -2))
        value = torch.cat([torch.sum(nf_games[:, ego, :] * joint_dist_mat, dim=(-1, -2)).unsqueeze(-1),
                           torch.sum(nf_games[:, partner, :] * joint_dist_mat, dim=(-1, -2)).unsqueeze(-1)], dim=1)
        return value
    def pareto(self, nf_games):
        batch_sz = nf_games.shape[0]
        all_dists = torch.zeros([batch_sz, self.num_agents, self.player_action_dim])
        all_values = torch.zeros(batch_sz, self.num_agents, device=self.device)
        # eye = torch.eye(self.player_action_dim, device=self.device)
        for i, g in enumerate(nf_games):
            sum_game = g.sum(dim=0)
            sum_max_val = torch.max(sum_game)
            action_idxs = (sum_game == sum_max_val).nonzero().squeeze(0)
            all_dists[i, 0, action_idxs[0]] = 1
            all_dists[i, 1, action_idxs[1]] = 1
            all_values[i, :] = g[:, action_idxs[0],action_idxs[1]]
        return all_dists,all_values

    def level_k_qunatal(self,nf_games, sophistication=4, belief_trick=True):
        """Implementes a k-bounded QRE computation
        https://en.wikipedia.org/wiki/Quantal_response_equilibrium
        as reationality -> infty, QRE -> Nash Equilibrium
        """
        rationality = self.rationality
        batch_sz = nf_games.shape[0]
        player_action_dim = nf_games.shape[2]
        uniform_dist = (torch.ones(batch_sz, player_action_dim,device=self.device)) / player_action_dim
        ego, partner = 0, 1

        def invert_game(g):
            "inverts perspective of the game"
            return torch.cat([torch.transpose(g[:, partner, :, :], -1, -2).unsqueeze(1),
                              torch.transpose(g[:, ego, :, :], -1, -2).unsqueeze(1)], dim=1)

        def softmax(x):
            return torch.softmax(x,dim=1)

        def step_QRE(game, k):
            if k == 0:  partner_dist = uniform_dist
            else:  partner_dist = step_QRE(invert_game(game), k - 1)

            Exp_qAi = torch.bmm(game[:, ego, :, :], partner_dist.unsqueeze(-1)).squeeze(-1)
            return softmax(rationality * Exp_qAi)

        dist1 = step_QRE(nf_games, sophistication)
        if belief_trick:
            # uses dist1 as partner belief prior for +1 sophistication
            Exp_qAi = torch.bmm(invert_game(nf_games)[:, ego, :, :],dist1.unsqueeze(-1)).squeeze(-1)
            dist2 = softmax(rationality * Exp_qAi)
        else: # recomputes dist2 from scratch
            dist2 = step_QRE(invert_game(nf_games), sophistication)
        dist = torch.cat([dist1.unsqueeze(1), dist2.unsqueeze(1)], dim=1)
        # joint_dist_mat = torch.bmm(dist1.unsqueeze(-1), torch.transpose(dist2.unsqueeze(-1), -1, -2))

        value = self.get_expected_equilibrium_value(nf_games, dist)
        return dist, value

    def compute_EQ(self, NF_Games, update=False):
        NF_Games = NF_Games.reshape(-1, self.num_agents, self.player_action_dim, self.player_action_dim)
        all_joint_actions = []

        # Compute equilibrium for each game
        if self.eq_sol == 'QRE': all_dists,all_ne_values = self.level_k_qunatal(NF_Games)
        # elif self.eq_sol == 'scaling_QRE': all_dists,all_ne_values = self.level_k_qunatal(NF_Games,scaling=True)
        elif self.eq_sol == 'Pareto': all_dists,all_ne_values = self.pareto(NF_Games)
        elif self.eq_sol == 'Nash': raise NotImplementedError # not feasible for gen. sum. game
        else: raise ValueError(f"Invalid EQ solution:{self.eq_sol}")

        if update:
            return all_dists, all_ne_values
        else:
            # Sample actions from strategies
            for _ in all_dists:
                a1, a2 = torch.multinomial(all_dists[0, :], 1).detach().cpu().numpy().flatten()
                action_idxs = (a1, a2)
                joint_action_idx = Action.INDEX_TO_ACTION_INDEX_PAIRS.index(action_idxs)
                all_joint_actions.append(joint_action_idx)
            return np.array(all_joint_actions), all_dists, all_ne_values

    def choose_joint_action(self, obs, epsilon=0.0, feasible_JAs= None, debug=False):
        sample = random.random()

        # Explore -------------------------------------
        if sample < epsilon:
            action_probs = np.ones(self.joint_action_dim) / self.joint_action_dim
            if feasible_JAs is not None:
                action_probs = feasible_JAs*action_probs
                action_probs = action_probs/np.sum(action_probs)
            joint_action_idx = np.random.choice(np.arange(self.joint_action_dim), p=action_probs)
            joint_action = self.joint_action_space[joint_action_idx]

        # Exploit -------------------------------------
        else:
            with torch.no_grad():
                NF_Game = self.get_normal_form_game(obs)
                joint_action_idx, dists, ne_vs = self.compute_EQ(NF_Game)
                joint_action_idx = joint_action_idx[0]
                joint_action = self.joint_action_space[joint_action_idx]
                action_probs = dists

        return joint_action, joint_action_idx, action_probs


    ###################################################
    # Update Utils ######################################
    ###################################################

    def update(self):
        # if (
        #         # self.memory_len < self.mem_size/2
        #         len(self._memory) < self._memory_batch_size
        # ):  return 0

        # transitions = self.memory_sample()
        transitions = self._memory.sample(self._memory_batch_size)

        batch = self._memory.transition(*zip(*transitions))
        BATCH_SIZE = len(transitions)
        state = torch.cat(batch.state)
        action = torch.cat(batch.action)
        reward = torch.cat(batch.reward)
        done = torch.cat(batch.done)
        #
        # # Q-Learning with target network
        q_value = self.model(state).gather(1, action)

        # Batch calculate Q(s'|pi) and form mask for later condensation to expectation
        # all_next_states,all_p_next_states,prospect_idxs = self.flatten_next_prospects(batch.next_prospects)
        all_next_states, all_p_next_states, prospect_idxs = flatten_next_prospects(batch.next_prospects)

        # Compute equalib value for each outcome ---------------
        # NF_games = self.get_normal_form_game(torch.cat(all_next_states), use_target=True)
        NF_games = self.get_normal_form_game(torch.cat(all_next_states), with_model=self.target)
        all_next_a_dists, all_next_q_value = self.compute_EQ(NF_games, update=True)

        # Convert to numpy then back ----------------------------
        all_next_q_value = all_next_q_value[:, 0].reshape(-1, 1).detach().cpu().numpy()
        all_p_next_states = np.array(all_p_next_states).reshape(-1, 1)

        expected_q_value = self.prospect_value_expectations(reward = reward,
                                                            done = done,
                                                            prospect_masks=prospect_idxs,
                                                            prospect_next_q_values= all_next_q_value,
                                                            prospect_p_next_states= all_p_next_states)

        # Optimize ----------------
        loss = F.mse_loss(q_value, expected_q_value.detach(), reduction='none')
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        if self.clip_grad is not None:
            # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()
        return loss.item()

    def prospect_value_expectations(self,reward,done,prospect_masks,prospect_next_q_values,prospect_p_next_states):
        """Rational expectation used for modification when class inherited by CPT version
        - condenses prospects back into expecations of |batch_size|
        """

        BATCH_SIZE = len(prospect_masks)
        # done = done.detach().cpu().numpy()
        # rewards = reward.detach().cpu().numpy()
        expected_td_targets = np.nan * np.ones([BATCH_SIZE, 1])
        for i in range(BATCH_SIZE):
            prospect_mask = prospect_masks[i]
            prospect_values = prospect_next_q_values[prospect_mask, :]
            prospect_probs = prospect_p_next_states[prospect_mask, :]
            prospect_td_targets = reward[i, :] + (self.gamma) * prospect_values * (1 - done[i, :])
            assert np.sum(prospect_probs) == 1, 'prospect probs should sum to 1'
            expected_td_targets[i] = np.sum(prospect_td_targets * prospect_probs)  # rational
        assert not np.any(np.isnan(expected_td_targets)), 'prospect expectations not filled'
        return torch.FloatTensor(expected_td_targets).to(self.device)



    def update_target(self):
        # self.target = soft_update(self.model, self.target, self.tau)
        target_net_state_dict = self.target.state_dict()
        policy_net_state_dict = self.model.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * (self.tau) + target_net_state_dict[key] * (1 - self.tau)
        self.target.load_state_dict(target_net_state_dict)
        return self.target

class SelfPlay_QRE_OSA_CPT(SelfPlay_QRE_OSA):
    def __init__(self, obs_shape, n_actions, agents_config, **kwargs):
        super().__init__(obs_shape, n_actions, agents_config, **kwargs)
        self.CPT = CumulativeProspectTheory(**agents_config['cpt'])

        self.frozen = False
        self.rational_ref_model = None

    def update(self):
        if self.frozen: raise ValueError('Model is frozen, cannot update')


        transitions = self._memory.sample(self._memory_batch_size)
        batch = self._memory.transition(*zip(*transitions))
        BATCH_SIZE = len(transitions)
        state = torch.cat(batch.state)
        action = torch.cat(batch.action)
        reward = np.vstack(batch.reward)
        # done = np.vstack(batch.done) # (excluded to solve infinite horizon)

        # # Q-Learning with target network
        q_value = self.model(state).gather(1, action)

        # Batch calculate Q(s'|pi) and form mask for later condensation to expectation
        all_next_states, all_p_next_states, prospect_idxs = flatten_next_prospects(batch.next_prospects)

        # Compute equalib value for each outcome ---------------
        with torch.no_grad():
            NF_games = self.get_normal_form_game(torch.cat(all_next_states), with_model=self.target)
            all_next_a_dists, all_next_q_value = self.compute_EQ(NF_games, update=True)

            # Convert to numpy then back ----------------------------
            all_next_q_value = all_next_q_value[:, 0].reshape(-1, 1).detach().cpu().numpy() # grab only ego values
            all_p_next_states = np.array(all_p_next_states).reshape(-1, 1)

            expected_value = self.CPT.expectation_samples(all_next_q_value, all_p_next_states,
                                                               prospect_idxs, reward, self.gamma)
            expected_value = torch.from_numpy(expected_value).float().cuda()

        # expected_value = torch.tensor(expected_value, dtype=torch.float32, device=self.device)
        # Optimize ----------------
        loss = F.mse_loss(q_value, expected_value.detach(), reduction='none')
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()
        return loss.item()


    # def prospect_value_expectations(self,reward,done,prospect_masks,
    #                                 prospect_next_q_values,prospect_p_next_states,
    #                                 prospect_next_q_values_ref = None,  debug=False):
    #     """CPT expectation used for modification when class inherited by CPT version
    #     - condenses prospects back into expecations of |batch_size|
    #     """
    #
    #     BATCH_SIZE = len(prospect_masks)
    #     # done = done.detach().cpu().numpy()
    #     # rewards = reward.detach().cpu().numpy()
    #     expected_td_targets = np.zeros([BATCH_SIZE, 1])
    #     # prospect_next_q_values = prospect_next_q_values.detach().cpu().numpy()
    #
    #     for i in range(BATCH_SIZE):
    #         prospect_mask = prospect_masks[i]
    #         prospect_values = prospect_next_q_values[prospect_mask, :]
    #         prospect_probs = prospect_p_next_states[prospect_mask, :]
    #         prospect_td_targets = reward[i, :] + (self.gamma) * prospect_values #* (1 - done[i, :])
    #
    #
    #         expected_td_targets[i] = self.CPT.OG_expectation(prospect_td_targets.flatten(),
    #                                                       prospect_probs.flatten())
    #
    #     # expected_td_targets = torch.tensor(expected_td_targets, dtype=torch.float32, device=self.device)
    #     return expected_td_targets



    # def prospect_value_expectations(self,reward,done,prospect_masks,
    #                                 prospect_next_q_values,prospect_p_next_states,
    #                                 prospect_next_q_values_ref = None,  debug=False):
    #     """CPT expectation used for modification when class inherited by CPT version
    #     - condenses prospects back into expecations of |batch_size|
    #     """
    #
    #     BATCH_SIZE = len(prospect_masks)
    #     # done = done.detach().cpu().numpy()
    #     # rewards = reward.detach().cpu().numpy()
    #     expected_td_targets = self.CPT.expectation_samples(prospect_next_q_values, prospect_p_next_states,
    #                                                        prospect_masks, reward, self.gamma)
    #     expected_td_targets = torch.tensor(expected_td_targets, dtype=torch.float32, device=self.device)
    #     return expected_td_targets


class DQN_vector_feature(nn.Module):

    def __init__(self, obs_shape, n_actions,num_hidden_layers,size_hidden_layers,**kwargs):
        self.num_hidden_layers = num_hidden_layers
        self.size_hidden_layers = size_hidden_layers
        self.mlp_activation = nn.LeakyReLU

        super(DQN_vector_feature, self).__init__()
        self.layer1 = nn.Linear(obs_shape[0], self.size_hidden_layers)
        self.layer2 = nn.Linear(self.size_hidden_layers, self.size_hidden_layers)
        self.layer3 = nn.Linear(self.size_hidden_layers, n_actions)

        layer_buffer = [ nn.Linear(obs_shape[0], self.size_hidden_layers),self.mlp_activation()]
        for i in range(1,self.num_hidden_layers-1):
            layer_buffer.extend([nn.Linear(self.size_hidden_layers, self.size_hidden_layers),self.mlp_activation()])
        layer_buffer.extend([nn.Linear(self.size_hidden_layers, n_actions)])

        self.layers = nn.Sequential(*layer_buffer)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        for module in self.layers:
            x = module(x)
        # x = self.mlp_activation(self.layer1(x))
        # x = self.mlp_activation(self.layer2(x))
        # x = self.layer3(x) # linear output layer (action-values)
        return x
