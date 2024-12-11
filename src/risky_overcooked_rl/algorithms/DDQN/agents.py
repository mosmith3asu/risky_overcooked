import random
import torch.optim.lr_scheduler as lr_scheduler

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
from risky_overcooked_rl.algorithms.DDQN.memory import ReplayMemory_Prospect,ReplayMemory_Simple
from risky_overcooked_rl.utils.risk_sensitivity import CumulativeProspectTheory
from risky_overcooked_rl.utils.model_manager import get_absolute_save_dir
from risky_overcooked_rl.utils.state_utils import invert_obs, invert_joint_action, invert_prospect
import numpy as np
import warnings
import copy
import os
plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")



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


class SelfPlay_QRE_OSA(object):
    @classmethod
    def from_file(cls,obs_shape, n_actions, config, fname):

        # instantiate base class -------------
        agents = cls(obs_shape, n_actions, config)

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
        loaded_model = torch.load(PATH, weights_only=True, map_location=config['device'])
        agents.model.load_state_dict(loaded_model)
        agents.target.load_state_dict(loaded_model)
        agents.checkpoint_model.load_state_dict(loaded_model)
        # is_same = np.all([torch.all(agents.model.state_dict()[key] == agents.model.state_dict()[key]) for key in
        #         agents.model.state_dict().keys()])
        return agents

    def __init__(self, obs_shape, n_actions, config,**kwargs):
        self.clip_grad = config['clip_grad']
        self.num_hidden_layers = config['num_hidden_layers']
        self.size_hidden_layers = config['size_hidden_layers']
        self.learning_rate = config['lr_sched'][1]
        self.device = config['device']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.eq_sol = 'QRE'
        # self.eq_sol = 'Pareto'
        self.rationality = 10
        self.num_agents = 2
        self.mem_size = config['replay_memory_size']
        self.joint_action_space = list(itertools.product(Action.ALL_ACTIONS, repeat=2))
        self.joint_action_dim = n_actions
        self.player_action_dim = int(np.sqrt(n_actions))

        # Define Memory
        self._memory = ReplayMemory_Prospect(self.mem_size,self.device)
        self._memory_batch_size = config['minibatch_size']
        # self._transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_prospects', 'done'))
        # self._memory = deque([], maxlen=self.mem_size)
        # self._memory_batch_size = config['minibatch_size']

        # Define Model
        self.model = DQN_vector_feature(obs_shape, n_actions,self.num_hidden_layers, self.size_hidden_layers).to(self.device)
        self.target = DQN_vector_feature(obs_shape, n_actions,self.num_hidden_layers, self.size_hidden_layers).to(self.device)
        self.checkpoint_model = DQN_vector_feature(obs_shape, n_actions,self.num_hidden_layers, self.size_hidden_layers).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.checkpoint_model.load_state_dict(self.model.state_dict())


        lr_warmup_iter = config['lr_sched'][2]
        lr_factor = config['lr_sched'][0]/config['lr_sched'][1]
        # lr_factor = self.lr_warmup_scale
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr_factor * self.learning_rate, amsgrad=True)
        self.scheduler = lr_scheduler.LinearLR(self.optimizer,
                                               start_factor=1,
                                               end_factor=1 / lr_factor,
                                               total_iters=lr_warmup_iter)
        self.optimistic_value_expectation = False
        if self.optimistic_value_expectation: warnings.warn("Optimistic value expectation is set to True.")

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

        if self.optimistic_value_expectation:
            value = torch.zeros(batch_sz, 2, device=self.device)
            pareto_vals = nf_games[:, partner, :] + nf_games[:, ego, :]
            for ib, pval in enumerate(pareto_vals):
                idx = (pval == torch.max(pval)).nonzero().flatten()
                value[ib, ego] += nf_games[ib, ego, idx[0], idx[1]]
                value[ib, partner] += nf_games[ib, partner, idx[0], idx[1]]
        else: value = self.get_expected_equilibrium_value(nf_games, dist)
            # value = torch.cat([torch.sum(nf_games[:, ego, :] * joint_dist_mat, dim=(-1, -2)).unsqueeze(-1),
            #          torch.sum(nf_games[:, partner, :] * joint_dist_mat, dim=(-1, -2)).unsqueeze(-1)],dim=1)
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
        all_next_states,all_p_next_states,prospect_idxs = self.flatten_next_prospects(batch.next_prospects)

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
        done = done.detach().cpu().numpy()
        rewards = reward.detach().cpu().numpy()
        expected_td_targets = np.nan * np.ones([BATCH_SIZE, 1])
        for i in range(BATCH_SIZE):
            prospect_mask = prospect_masks[i]
            prospect_values = prospect_next_q_values[prospect_mask, :]
            prospect_probs = prospect_p_next_states[prospect_mask, :]
            prospect_td_targets = rewards[i, :] + (self.gamma) * prospect_values * (1 - done[i, :])
            assert np.sum(prospect_probs) == 1, 'prospect probs should sum to 1'
            expected_td_targets[i] = np.sum(prospect_td_targets * prospect_probs)  # rational
        assert not np.any(np.isnan(expected_td_targets)), 'prospect expectations not filled'
        return torch.FloatTensor(expected_td_targets).to(self.device)

        ######### ORIGONAL ##############################
        # RESULTS IN ROUNDING ERROR DIFFERENT FROM CPT ##
        # Calculates expectation of Q', not target. #####
        # BATCH_SIZE = len(prospect_masks)
        # done = done.detach().cpu().numpy()
        # rewards = reward.detach().cpu().numpy()
        # expected_next_q_values = np.nan * np.ones([BATCH_SIZE, 1])
        # for i in range(BATCH_SIZE):
        #     prospect_mask = prospect_masks[i]
        #     prospect_values = prospect_next_q_values[prospect_mask, :]
        #     prospect_probs = prospect_p_next_states[prospect_mask, :]
        #     assert np.sum(prospect_probs) == 1, 'prospect probs should sum to 1'
        #     expected_next_q_values[i] = np.sum(prospect_values * prospect_probs)
        # assert not np.any(np.isnan(expected_next_q_values)), 'prospect expectations not filled'
        # # expected_next_q_values = torch.FloatTensor(expected_next_q_values).to(self.device)
        # expected_q_value = rewards + (self.gamma) * expected_next_q_values * (1 - done)  # TD-Target
        # return torch.FloatTensor(expected_q_value).to(self.device)

    def flatten_next_prospects(self,next_prospects):
        """
        Used for flattening next_state prospects into list of outcomes for batch processing
         - improve model-value prediction speed
         - condensed to back to |batch_size| after using expectation
        """

        all_next_states = []
        all_p_next_states = []
        prospect_idxs = []
        total_outcomes = 0
        for i, prospect in enumerate(next_prospects):
            n_outcomes = len(prospect)
            all_next_states += [outcome[1] for outcome in prospect]
            all_p_next_states += [outcome[2] for outcome in prospect]
            prospect_idxs.append(np.arange(total_outcomes, total_outcomes + n_outcomes))
            total_outcomes += n_outcomes
        return all_next_states,all_p_next_states,prospect_idxs

    def update_target(self):
        # self.target = soft_update(self.model, self.target, self.tau)
        target_net_state_dict = self.target.state_dict()
        policy_net_state_dict = self.model.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * (self.tau) + target_net_state_dict[key] * (1 - self.tau)
        self.target.load_state_dict(target_net_state_dict)
        return self.target

class SelfPlay_QRE_OSA_CPT(SelfPlay_QRE_OSA):
    def __init__(self, obs_shape, n_actions, config, **kwargs):
        super().__init__(obs_shape, n_actions, config, **kwargs)
        self.CPT = CumulativeProspectTheory(**config['cpt_params'])

        self.frozen = False

        if self.CPT.exp_rational_value_ref == True:
            assert 'rational' in config['loads'], 'Rational reference needs loaded model'
            self.rational_ref_model = DQN_vector_feature(obs_shape, n_actions, self.num_hidden_layers, self.size_hidden_layers).to(self.device)
            # self.rational_ref_model = SelfPlay_QRE_OSA.from_file(obs_shape, n_actions, config).model
            self.rational_ref_model.load_state_dict(self.model.state_dict())        # Define Model
            self.rational_ref_model.eval() # permanently set to eval mode

        else: self.rational_ref_model = None

    def update(self):
        if self.frozen: raise ValueError('Model is frozen, cannot update')

        # if (
        #         len(self._memory) < self.mem_size
        #         # len(self._memory) < self.mem_size / 2
        #         # self.memory_len < self.mem_size / 2
        #         # self.memory_len < self._memory_batch_size
        #         # or self.memory_len < 0.25*self.mem_size
        # ):  return 0
        if self._memory.with_priority:
            transitions, weights, tree_idxs = self._memory.priority_sample(self._memory_batch_size)
            weights = weights.to(self.device)
        else:
            transitions = self._memory.sample(self._memory_batch_size)
            BATCH_SIZE = len(transitions)
            weights = torch.ones([BATCH_SIZE,1],device=self.device)
        # transitions = self._memory.sample(self._memory_batch_size)
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
        all_next_states, all_p_next_states, prospect_idxs = self.flatten_next_prospects(batch.next_prospects)

        # Compute equalib value for each outcome ---------------
        NF_games = self.get_normal_form_game(torch.cat(all_next_states), with_model=self.target)
        all_next_a_dists, all_next_q_value = self.compute_EQ(NF_games, update=True)

        # Convert to numpy then back ----------------------------
        all_next_q_value = all_next_q_value[:, 0].reshape(-1, 1).detach().cpu().numpy() # grab only ego values
        all_p_next_states = np.array(all_p_next_states).reshape(-1, 1)

        # IF using rational reference, get the rational expectations of following equalib solution
        # if self.CPT.exp_rational_value_ref:
        #     with torch.no_grad():
        #         NF_games_ref = self.get_normal_form_game(torch.cat(all_next_states), with_model=self.rational_ref_model)
        #         all_next_value_ref = self.get_expected_equilibrium_value(NF_games_ref, all_next_a_dists)
        #         all_next_value_ref = all_next_value_ref[:, 0].reshape(-1, 1).detach().cpu().numpy()
        # else: all_next_value_ref = None
        all_next_value_ref = None


        expected_q_value = self.prospect_value_expectations(reward=reward,
                                                            done=done,
                                                            prospect_masks=prospect_idxs,
                                                            prospect_next_q_values=all_next_q_value,
                                                            prospect_p_next_states=all_p_next_states,
                                                            prospect_next_q_values_ref = all_next_value_ref)

        # Optimize ----------------
        if self._memory.with_priority:
            TD_error = torch.abs(q_value - expected_q_value)
            self._memory.update_priorities(tree_idxs, TD_error.detach().cpu().numpy())
            loss = torch.mean((q_value - expected_q_value) ** 2 * weights)
        else:
            loss = F.mse_loss(q_value, expected_q_value.detach(), reduction='none')
            loss = loss.mean()
        # loss = F.mse_loss(q_value, expected_q_value.detach(), reduction='none')
        # loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        if self.clip_grad is not None:
            # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()
        return loss.item()
    def prospect_value_expectations(self,reward,done,prospect_masks,
                                    prospect_next_q_values,prospect_p_next_states,
                                    prospect_next_q_values_ref = None,  debug=False):
        """CPT expectation used for modification when class inherited by CPT version
        - condenses prospects back into expecations of |batch_size|
        """

        BATCH_SIZE = len(prospect_masks)
        done = done.detach().cpu().numpy()
        rewards = reward.detach().cpu().numpy()
        expected_td_targets = np.zeros([BATCH_SIZE, 1])
        for i in range(BATCH_SIZE):
            prospect_mask = prospect_masks[i]
            prospect_values = prospect_next_q_values[prospect_mask, :]
            prospect_probs = prospect_p_next_states[prospect_mask, :]
            prospect_td_targets = rewards[i, :] + (self.gamma) * prospect_values * (1 - done[i, :])
            if  prospect_next_q_values_ref is not None:
                prospect_values_ref = prospect_next_q_values_ref[prospect_mask, :]
                prospect_td_targets_ref = rewards[i, :] + (self.gamma) * prospect_values_ref * (1 - done[i, :])
                prospect_td_targets_ref = prospect_td_targets_ref.flatten()
            else: prospect_td_targets_ref = None
            if debug: assert np.sum(prospect_probs) == 1, 'prospect probs should sum to 1'

            expected_td_targets[i] = self.CPT.expectation(prospect_td_targets.flatten(),
                                                          prospect_probs.flatten(),
                                                          value_refs = prospect_td_targets_ref)
            if debug and self.CPT.is_rational:
                rat_expected_td_target = np.sum(prospect_td_targets * prospect_probs)
                # assert np.all(rat_expected_td_target == expected_td_targets[i]), \
                #     'Rational CPT expectation not equal to sum of prospect values'
                assert np.all(np.isclose(rat_expected_td_target,expected_td_targets[i])),\
                    'Rational CPT expectation not equal to sum of prospect values'
        expected_td_targets = torch.tensor(expected_td_targets, dtype=torch.float32, device=self.device)
        return expected_td_targets


class ResponseAgent(object):

    def __init__(self, obs_shape, n_actions, config,cpt_agent,
                 use_partner_prior=True,**kwargs):
        self.clip_grad = config['clip_grad']
        self.num_hidden_layers = config['num_hidden_layers']
        self.size_hidden_layers = config['size_hidden_layers']
        self.learning_rate = config['lr_sched'][1]
        self.device = config['device']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.eq_sol = 'QRE'
        # self.eq_sol = 'Pareto'
        self.rationality = 10
        self.num_agents = 2
        self.mem_size = config['replay_memory_size']
        self.joint_action_space = list(itertools.product(Action.ALL_ACTIONS, repeat=2))
        self.joint_action_dim = n_actions
        self.player_action_dim = int(np.sqrt(n_actions))
        self.cpt_agent = cpt_agent

        # Define Memory
        self._memory = ReplayMemory_Simple(self.mem_size,self.device)
        self._memory_batch_size = config['minibatch_size']

        # Define Model
        self.model = DQN_vector_feature(obs_shape, n_actions,self.num_hidden_layers, self.size_hidden_layers).to(self.device)
        self.target = DQN_vector_feature(obs_shape, n_actions,self.num_hidden_layers, self.size_hidden_layers).to(self.device)
        self.checkpoint_model = DQN_vector_feature(obs_shape, n_actions,self.num_hidden_layers, self.size_hidden_layers).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.checkpoint_model.load_state_dict(self.model.state_dict())
        if use_partner_prior:
            self.load_agent_dict(cpt_agent)


        lr_warmup_iter = config['lr_sched'][2]
        lr_factor = config['lr_sched'][0]/config['lr_sched'][1]
        # lr_factor = self.lr_warmup_scale
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr_factor * self.learning_rate, amsgrad=True)
        self.scheduler = lr_scheduler.LinearLR(self.optimizer,
                                               start_factor=1,
                                               end_factor=1 / lr_factor,
                                               total_iters=lr_warmup_iter)
        self.optimistic_value_expectation = False
        if self.optimistic_value_expectation: warnings.warn("Optimistic value expectation is set to True.")

    def load_agent_dict(self,agent):
        self.model.load_state_dict(agent.model.state_dict())
        self.target.load_state_dict(agent.model.state_dict())
        self.checkpoint_model.load_state_dict(agent.model.state_dict())


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
            if i == 1:
                obs = invert_obs(obs)
                q_values = self.cpt_agent.model(obs).detach()
                q_values = torch.transpose(q_values, -1, -2)
            elif with_model is not None: q_values = with_model(obs).detach()
            else:  q_values = self.model(obs).detach()
            q_values = q_values.reshape(batch_size, self.player_action_dim, self.player_action_dim)
            all_games[:, i, :, :] = q_values
        return all_games

    def get_expected_equilibrium_value(self, nf_games, dists):
        ego, partner = 0, 1
        joint_dist_mat = torch.bmm(dists[:,ego].unsqueeze(-1), torch.transpose(dists[:,partner].unsqueeze(-1), -1, -2))
        value = torch.cat([torch.sum(nf_games[:, ego, :] * joint_dist_mat, dim=(-1, -2)).unsqueeze(-1),
                           torch.sum(nf_games[:, partner, :] * joint_dist_mat, dim=(-1, -2)).unsqueeze(-1)], dim=1)
        return value

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

        if self.optimistic_value_expectation:
            value = torch.zeros(batch_sz, 2, device=self.device)
            pareto_vals = nf_games[:, partner, :] + nf_games[:, ego, :]
            for ib, pval in enumerate(pareto_vals):
                idx = (pval == torch.max(pval)).nonzero().flatten()
                value[ib, ego] += nf_games[ib, ego, idx[0], idx[1]]
                value[ib, partner] += nf_games[ib, partner, idx[0], idx[1]]
        else: value = self.get_expected_equilibrium_value(nf_games, dist)
            # value = torch.cat([torch.sum(nf_games[:, ego, :] * joint_dist_mat, dim=(-1, -2)).unsqueeze(-1),
            #          torch.sum(nf_games[:, partner, :] * joint_dist_mat, dim=(-1, -2)).unsqueeze(-1)],dim=1)
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
            # action_probs = np.ones(self.joint_action_dim) / self.joint_action_dim
            # if feasible_JAs is not None:
            #     action_probs = feasible_JAs*action_probs
                # action_probs = action_probs/np.sum(action_probs)
            # joint_action_idx = np.random.choice(np.arange(self.joint_action_dim), p=action_probs)
            # joint_action = self.joint_action_space[joint_action_idx]
            action_probs = np.ones(len(Action.ALL_ACTIONS)) / len(Action.ALL_ACTIONS)
            ego_action_idx = np.random.choice(np.arange(len(Action.ALL_ACTIONS)), p=action_probs)
            ego_action = Action.ALL_ACTIONS[ego_action_idx]
            partner_action = self.cpt_agent.choose_joint_action(obs, epsilon=0)[0][1]
            joint_action = (ego_action, partner_action)
            joint_action_idx = Action.ALL_JOINT_ACTIONS.index(joint_action)
            action_probs = None

        # Exploit -------------------------------------
        else:
            with torch.no_grad():
                NF_Game = self.get_normal_form_game(obs)
                joint_action_idx, dists, ne_vs = self.compute_EQ(NF_Game)
                joint_action_idx = joint_action_idx[0]
                joint_action = self.joint_action_space[joint_action_idx]
                action_probs = dists

                # Check feasible actions and resample
                # if feasible_JAs is not None:
                #     na = len(Action.ALL_ACTIONS)
                #     feasibleM = feasible_JAs.reshape([na, na])
                #     feasible_As = np.array([[np.any(feasibleM[ia,:]) for ia in range(6)],
                #                             [np.any(feasibleM[:,ia]) for ia in range(6)]])
                #     action_probs = feasible_As * action_probs.detach().cpu().numpy()[0]
                #     action_idxs = [np.random.choice(np.arange(self.player_action_dim),
                #                                     p=action_probs[ip]/action_probs[ip].sum())
                #                          for ip in range(2)]
                #     joint_action_idx = Action.INDEX_TO_ACTION_INDEX_PAIRS.index(tuple(action_idxs))
                #     joint_action = self.joint_action_space[joint_action_idx]


        return joint_action, joint_action_idx, action_probs


    ###################################################
    # Update Utils ######################################
    ###################################################

    def update(self):
        if self._memory.with_priority and  len(self._memory) < self.mem_size:
            return 0
        # elif (
        #         # self.memory_len < self.mem_size/2
        #         len(self._memory) < self._memory_batch_size
        # ):  return 0

        ego, partner = 0,1
        if self._memory.with_priority:
            transitions, weights, tree_idxs = self._memory.priority_sample(self._memory_batch_size)
            weights = weights.to(self.device)
        else:
            transitions = self._memory.sample(self._memory_batch_size)
            BATCH_SIZE = len(transitions)
            weights = torch.ones([BATCH_SIZE,1],device=self.device)

        batch = self._memory.transition(*zip(*transitions))

        state = torch.cat(batch.state)
        action = torch.cat(batch.action)
        reward = torch.cat(batch.reward)
        next_state = torch.cat(batch.next_state)
        done = torch.cat(batch.done)

        # # Q-Learning with target network
        q_value = self.model(state).gather(1, action)

        # Batch calculate Q(s'|pi) and form mask for later condensation to expectation
        # all_next_states,all_p_next_states,prospect_idxs = self.flatten_next_prospects(batch.next_prospects)

        # Compute equalib value for each outcome ---------------
        # NF_games = self.get_normal_form_game(torch.cat(all_next_states), use_target=True)
        # NF_games = self.get_normal_form_game(torch.cat(all_next_states), with_model=self.target)
        NF_games = self.get_normal_form_game(next_state, with_model=self.target)
        _, next_q_value = self.compute_EQ(NF_games, update=True)

        # Convert to numpy then back ----------------------------
        expected_q_value = reward + (self.gamma) * next_q_value[:,ego].unsqueeze(-1) * (1 - done)

        # Optimize ----------------
        if self._memory.with_priority:
            TD_error = torch.abs(q_value - expected_q_value)
            self._memory.update_priorities(tree_idxs, TD_error.detach().cpu().numpy())
            loss = torch.mean((q_value - expected_q_value) ** 2 * weights)
        else:
            loss = F.mse_loss(q_value, expected_q_value.detach(), reduction='none')
            loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad is not None:
            # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        # self.target = soft_update(self.model, self.target, self.tau)
        target_net_state_dict = self.target.state_dict()
        policy_net_state_dict = self.model.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * (self.tau) + target_net_state_dict[key] * (1 - self.tau)
        self.target.load_state_dict(target_net_state_dict)
        return self.target

