import itertools
import warnings

import numpy as np
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedState
from risky_overcooked_py.mdp.actions import Action, Direction
from risky_overcooked_py.agents.agent import Agent, AgentPair,StayAgent, RandomAgent, GreedyHumanModel
import pickle
from risky_overcooked_rl.utils.deep_models import ReplayMemory,DQN_vector_feature
import nashpy as nash
import os

from collections import Counter, defaultdict
import torch
from itertools import product,count
import random

class SoloDeepQAgent(Agent):
    """An agent randomly picks motion actions.
    Note: Does not perform interat actions, unless specified"""

    def __init__(self, mdp, agent_index, policy_net, # model,
                 save_agent_file=False, load_learned_agent=False,
                 config=None,verbose_load_config=True):
        super(SoloDeepQAgent, self).__init__()


        # Featurize Params ---------------
        self.my_index = agent_index
        self.partner_index = int(not self.my_index)
        self.N_PLAYER_FEAT = 9 # for each player; used for inverse observation

        # Create learning environment ---------------
        self.mdp = mdp
        if config['n_actions'] == 6:    self.action_space = Action.ALL_ACTIONS
        elif config['n_actions'] == 36: self.action_space = list(itertools.product(Action.ALL_ACTIONS, repeat=2)) # CLCE
        else: raise ValueError(f"Invalid number of actions: {config['n_actions']} != 6 or 36")


        if config is not None: self.load_config(config,verbose=verbose_load_config)
        else: warnings.warn("No config provided, using default values. Please ensure defaults are correct")

        self.policy_net = policy_net
        self.device = config['device']

        # self.model = model(**config)  # DQN model used for learning

    def load_config(self, config,verbose=True):
        if verbose:
            print(f'--------------------------------------------------')
            print(f"Loading config for agent {self.my_index}:")
            print(f'--------------------------------------------------')
        for key, value in config.items():
            setattr(self, key, value)
            if verbose:  print(f"\t|{key}: {value}")
        if verbose:
            print(f'--------------------------------------------------')
    def print_config(self):
        report_list = ['exploration_proba', 'max_exploration_proba', 'min_exploration_proba', 'gamma', 'lr', 'rationality']
        print(f'--------------------------------------------------')
        print(f'Agent {self.my_index} Config:')
        print(f'--------------------------------------------------')
        for key in report_list:
            print(f"\t|{key}: {getattr(self, key)}")
        print(f'--------------------------------------------------\n')
    ########################################################
    # Learning/Performance methods #########################
    ########################################################
    def action(self, state, exp_prob=0, rationality='max',debug=False):
        """
        :param state: OvercookedState object
        :param exp_prob: probability of random exploration {0: always exploit, 1: always explore}
        :param rationality: if not exploring, Boltzmann rationality temperature ('max' if argmax)
        :return: action, action_info = {"action_index": int, "action_probs": np.ndarray}
        """
        # global steps_done
        sample = random.random()

        # EXPLORE ----------------
        if sample < exp_prob:
            action_probs = np.ones(len(self.action_space)) / len(self.action_space)
            ai = np.random.choice(np.arange(len(self.action_space)), p=action_probs)
            action = self.action_space[ai]

        # EXPLOIT ----------------
        else:
            obs = self.featurize(state)
            with torch.no_grad():
                if rationality == 'max':
                    ai = self.policy_net(obs).max(1).indices.view(1, 1).numpy().flatten()[0]
                    action_probs = np.eye(len(self.action_space))[ai]
                    action = self.action_space[ai]
                else:
                    qA = self.policy_net(obs).numpy().flatten()
                    action_probs = self.softmax(rationality * qA)
                    ai = np.random.choice(self.action_space, p=action_probs)
                    action = self.action_space[ai]
        # RETURN ----------------
        action_info = {"action_index": ai, "action_probs": action_probs}
        return action, action_info

    def actions(self, states, agent_indices):
        return (self.action(state) for state in states)

    def softmax(self, x):
        e_x = np.exp(x- np.max(x))
        return e_x / e_x.sum()

    def update(self, transitions,GAMMA):
        """ Provides batch update to the DQN model """
        # optimize_model(self.policy_net, self.target_net, self.optimizer, transitions, GAMMA)
        raise NotImplementedError

    def valuation_fun(self, td_target):
        """ Applies CPT (if specified) to the target Q values"""
        return td_target

    ########################################################
    # Featurization methods ################################
    ########################################################
    def featurize(self, overcooked_state,as_tensor=True):
        """ Generate a feature vector"""
        obs = self.mdp.get_lossless_encoding_vector(overcooked_state)
        if as_tensor: obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.my_index == 1:  obs = self.reverse_state_obs(obs)
        return obs

    def get_featurized_shape(self):
        """ Get the shape of either type of the featurized state"""
        return self.mdp.get_lossless_encoding_vector_shape()

    def handle_state_obs_type(self,state):
        """change ambigous type of state/obs into tensor obs encoding"""
        if isinstance(state,OvercookedState): # is an overcooked state object
            return self.featurize(state)
        elif isinstance(state,np.ndarray): # is an encoding vector but np array
            return torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        elif isinstance(state,torch.Tensor): # already a encoding tensor
            return state
        else: raise ValueError(f'Unknown state type: {type(state)}')

    def reverse_state_obs(self,obs):
        obs = torch.cat([obs[:, self.N_PLAYER_FEAT:2 * self.N_PLAYER_FEAT],
                         obs[:, :self.N_PLAYER_FEAT],
                         obs[:, 2 * self.N_PLAYER_FEAT:]], dim=1)
        return obs

    ########################################################
    # Saving utils #########################################
    ########################################################
    def save_agent(self, filename):
        """
        Save the Q-table to a file.

        Args:
            filename (str): The name of the file to save the Q-table to.

        Returns:
            None
        """
        # filename = os.path.join(os.path.dirname(__file__), filename)
        filename = os.path.join(self.learned_agent_path, filename)
        with open(filename, 'wb') as f:
            pickle.dump(self.Q_table, f)

    def load_agent(self, filename):
        """
        Load the Q-table from a file.

        Args:
            filename (str): The name of the file to load the Q-table from.

        Returns:
            None
        """
        filename = os.path.join(os.path.dirname(__file__), filename)
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)


class SelfPlay_DeepAgentPair(object):
    def __init__(self,agent1,agent2,equalib='nash'):
        super(SelfPlay_DeepAgentPair, self).__init__()
        self.agents = [agent1,agent2]
        # assert equalib.lower() in ['nash','pareto','qre'], 'Unknown Equlibrium'
        assert equalib.lower() in ['nash', 'pareto'] or 'qre' in equalib.lower(), 'Unknown Equlibrium'
        self.equalib = equalib                      # equalibrium solution
        self.mdp = agent1.mdp                       # same for both agents
        self.device = agent1.device                 # same for both agents
        self.N_PLAYER_FEAT = agent1.N_PLAYER_FEAT   # same for both agents
        self.my_index = -1 # used for interfacing with Agent class utils


        self.action_space = list(itertools.product(Action.ALL_ACTIONS, repeat=2))

        # Grab methods from other class (may induce errors w. agent index?)
        self.featurize = agent1.featurize
        self.reverse_state_obs = agent1.reverse_state_obs
        self.handle_state_obs_type = agent1.handle_state_obs_type


    # Joint action ########################################
    def action(self, state, exp_prob=0, rationality='max', debug=False):
        """
        :param state: OvercookedState object
        :return: joint action for each agent
        """
        # global steps_done
        sample = random.random()

        # EXPLORE ----------------
        if sample < exp_prob:
            action_probs = np.ones(len(self.action_space)) / len(self.action_space)
            joint_action_idx = np.random.choice(np.arange(len(self.action_space)), p=action_probs)
            action = self.action_space[joint_action_idx]

        # EXPLOIT ----------------
        else:
            obs = self.handle_state_obs_type(state) # make correct type

            # Get normal form game
            flattened_game = np.zeros((2,len(self.action_space)))
            for ip,agent in enumerate(self.agents):
                if ip == 1: obs = self.reverse_state_obs(obs) # make into ego observation
                # obs = agent.featurize(state) # TODO: super inefficient to recalc this instead of inverting
                with torch.no_grad():
                    qAA = agent.policy_net(obs).numpy().flatten() #quality of joint actions
                    flattened_game[ip,:] = qAA
            NF_game = flattened_game.reshape([2, 6,6]) # normal form bi-matrix game

            # Apply equlibrium solution
            if self.equalib.lower() == 'nash':      action_idxs = self.nash(NF_game)
            elif self.equalib.lower() == 'pareto':  action_idxs = self.pareto(NF_game)
            elif 'qre' in self.equalib.lower() :    action_idxs = self.quantal_response(NF_game)
            else: raise ValueError(f'Unknown equalibrium: {self.equalib}')
            action = [Action.INDEX_TO_ACTION[ai] for ai in action_idxs]
            joint_action_idx = Action.INDEX_TO_ACTION_INDEX_PAIRS.index(action_idxs)
            action_probs = None

        # RETURN ----------------
        action_info = {"action_index": joint_action_idx, "action_probs": action_probs}
        return action, action_info

    ########################################################
    # Equilibrium ########################################
    ########################################################
    def quantal_response(self,game,soph=2):
        """quantal response equalibrium with k-order ToM (sophistication)"""
        def invert_game(g):
            "inverts perspective of the game"
            return np.array([g[1, :].T, g[0, :].T])
        def QRE(game, k):
            """recursive function resolving k-order ToM"""
            if k == 0: partner_dist = (np.arange(np.shape(game)[1])).reshape(1, np.shape(game)[1])
            else: partner_dist = QRE(invert_game(game), k - 1)
            weighted_game = game[0] * partner_dist
            Exp_qAi = np.sum(weighted_game, axis=1)
            return softmax(Exp_qAi)
        # check if last character in self.equalib is an integer
        if self.equalib[-1].isnumeric():  # if sophistication is specified
            soph = int(self.equalib[-1])
        na = np.shape(game)[1]
        pdA1 = QRE(game,k=soph)
        pdA2 = QRE(invert_game(game),k=soph)
        a1 = np.random.choice(np.arange(na),p=pdA1)
        a2 = np.random.choice(np.arange(na),p=pdA2)
        action_idxs = (a1,a2)
        return action_idxs


    def nash(self,game):
        na = np.shape(game)[1]
        rps = nash.Game(game[0,:], game[1,:])

        # find all equalibriums
        # eqs = list(rps.support_enumeration())
        eqs = list(rps.vertex_enumeration())

        # select equalibrium with highest pareto efficiency
        pareto_efficiencies = [np.sum(rps[eq[0], eq[1]]) for eq in eqs]
        best_eq_idx = pareto_efficiencies.index(max(pareto_efficiencies))
        mixed_eq = np.abs(eqs[best_eq_idx])

        # Sample actions according to equalib
        a1 = np.random.choice(np.arange(na), p=mixed_eq[0])
        a2 = np.random.choice(np.arange(na), p=mixed_eq[1])
        action_idxs = (a1,a2)
        return action_idxs

    def pareto(self,game):
        """
        :param game: normal form game [n_agents, |Ai|,|Aj|]
        :return:
        """
        cum_payoff_matrix = np.sum(game,axis=0)
        action_idxs = np.unravel_index(cum_payoff_matrix.argmax(), cum_payoff_matrix.shape)
        return action_idxs


    # def update(self,policy_net, target_net, optimizer, transitions, GAMMA):
    #     """Could grab policy and target net from the agents"""
    #
    #
    #     optimize_model(policy_net, target_net, optimizer, transitions, GAMMA)
    #




def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
















# def test_featurize():
#     config = {
#         "featurize_fn": "mask",
#         "horizon": 500
#     }
#     from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld
#     from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
#
#     LAYOUT = "sanity_check_3_onion"; HORIZON = 500; ITERATIONS = 10_000
#
#     # Generate MDP and environment----------------
#     # mdp_gen_params = {"layout_name": 'cramped_room_one_onion'}
#     # mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_params)
#     # env = OvercookedEnv(mdp_fn, horizon=HORIZON)
#     base_mdp = OvercookedGridworld.from_layout_name(LAYOUT)
#     base_env = OvercookedEnv.from_mdp(base_mdp, horizon=HORIZON)
#
#     # Generate agents
#     q_agent = SoloDeepQAgent(base_mdp,agent_index=0, save_agent_file=True,config=config)
#     stay_agent = StayAgent()
#     # agents = [CustomQAgent(mdp, is_learning_agent=True, save_agent_file=True), StayAgent()]
#
#     base_env.reset()
#     obs = q_agent.featurize(base_env.state)
#     print(f'Featurized Shape:{q_agent.get_featurized_shape()}')
#     print(f'Featurized Obs: {np.shape(obs)}:') # {obs}
#
#     # print(f'Q-table Shape: {np.shape(q_agent.Q_table)}')
#     # q_agent.Q_table[tuple(obs)][0] = 1
#     # q = q_agent.Q_table[tuple(obs)]
#
#     # print(f'Q-val ({np.shape(q)}): {q}')
#     # Get the index of obs in the Q-table
#
#
# if __name__ == "__main__":
#     # test_featurize()
#     test_update()
#
#     #
#     #
    #
# def verify_lossless_featurization(mdp, agent,state):
#     """ Verify that the lossless featurization is correct"""
#     primary_agent_idx = 0
#     other_agent_idx = 1
#     # Get baseline layer dict
#     # ordered_player_features = ["player_{}_loc".format(primary_agent_idx),
#     #                            "player_{}_loc".format(other_agent_idx)] + \
#     #                           ["player_{}_orientation_{}".format(i, Direction.DIRECTION_TO_INDEX[d])
#     #                            for i, d in product([primary_agent_idx, other_agent_idx],
#     #                                                          Direction.ALL_DIRECTIONS)]
#     #
#     # base_map_features = ["pot_loc", "counter_loc", "onion_disp_loc", "dish_disp_loc", "serve_loc"]
#     # variable_map_features = ["onions_in_pot", "onions_cook_time", "onion_soup_loc", "dishes", "onions"]
#     # urgency_features = ["urgency"]
#     # ordered_player_features = [
#     #                               "player_{}_loc".format(primary_agent_idx),
#     #                               "player_{}_loc".format(other_agent_idx),
#     #                           ] + [
#     #                               "player_{}_orientation_{}".format(i, Direction.DIRECTION_TO_INDEX[d])
#     #                               for i, d in product(
#     #         [primary_agent_idx, other_agent_idx],
#     #         Direction.ALL_DIRECTIONS,
#     #     )
#     #                           ]
#     # base_map_features = [
#     #     "pot_loc",
#     #     "counter_loc",
#     #     "onion_disp_loc",
#     #     "tomato_disp_loc",
#     #     "dish_disp_loc",
#     #     "serve_loc",
#     #     "water_loc"
#     # ]
#     # variable_map_features = [
#     #     "onions_in_pot",
#     #     "tomatoes_in_pot",
#     #     "onions_in_soup",
#     #     "tomatoes_in_soup",
#     #     "soup_cook_time_remaining",
#     #     "soup_done",
#     #     "dishes",
#     #     "onions",
#     #     "tomatoes",
#     # ]
#     # urgency_features = ["urgency"]
#     # baseline_LAYERS = ordered_player_features  + base_map_features + variable_map_features + urgency_features
#     baseline_LAYERS = ['player_0_loc', 'player_1_loc', 'player_0_orientation_0', 'player_0_orientation_1', 'player_0_orientation_2',
#      'player_0_orientation_3', 'player_1_orientation_0', 'player_1_orientation_1', 'player_1_orientation_2',
#      'player_1_orientation_3', 'pot_loc', 'counter_loc', 'onion_disp_loc', 'tomato_disp_loc', 'dish_disp_loc',
#      'serve_loc', 'onions_in_pot', 'tomatoes_in_pot', 'onions_in_soup', 'tomatoes_in_soup', 'soup_cook_time_remaining',
#      'soup_done', 'dishes', 'onions', 'tomatoes', 'urgency']
#
#     # for name in baseline_LAYERS:
#     #     print(f'LAYERS: {name}')
#     # print(f'LAYERS Len: {len(baseline_LAYERS)}')
#
#     # Get the lossless state encoding
#     obs_baseline = mdp.lossless_state_encoding(state)[0]
#     # Get the lossless state encoding from the agent
#     obs_agent,obs_agent_keys = agent.featurize_mask(state,get_keys=True)
#
#     # Check if all layers are present and correct
#     # for i,layer in enumerate(baseline_LAYERS):
#     #     if layer not in obs_agent_keys:
#     #         print(f'lAYE not in Custom: {layer}')
#     #         # raise AssertionError(f'Key not in baseline: {key}')
#     #     else:
#     #         i_agent = np.where(np.array(obs_agent_keys)==layer)[0][0]
#     #         is_same = np.all(obs_baseline[...,i]==obs_agent[...,i_agent])
#     #         print(f'Layer Check: [{layer} {obs_agent_keys[i_agent]}: {is_same}')
#     #         if not is_same:
#     #             print(f'Baseline:\n {obs_baseline[...,i]}')
#     #             print(f'Obs:\n {obs_agent[...,i_agent]}')
#
#
#     # Check if all are bool
#     layer_names = [baseline_LAYERS,obs_agent_keys]
#     for j,obs in enumerate([obs_baseline,obs_agent]):
#         if not np.all(np.array(obs) <= 1):
#             for i in range(obs.shape[-1]):
#                 a = obs[..., i]
#                 if not np.all(a <= 1) and not layer_names[j][i] in ['soup_cook_time_remaining','onions_in_pot','onions_in_soup']:
#                     print(f'(MDP) Non-bool lossless encoding {np.max(a)} @ feature {i}:{baseline_LAYERS[i]}')
#                     print(np.transpose(obs[..., i]))
#                     raise AssertionError(f'({["MDP","Agent"][j]}) Non-bool lossless encoding {np.max(a)} @ feature {i}:{layer_names[j][i]}')
#     # assert False
#     return True
# def test_update(n_tests = 100):
#     model_config = {
#         "obs_shape": [18, 5, 3],
#         "n_actions": 6,
#         "num_filters": 25,
#         "num_convs": 3,  # CNN params
#         "num_hidden_layers": 3,
#         "size_hidden_layers": 32,
#         "learning_rate": 1e-3,
#         "n_mini_batch": 6,
#         "minibatch_size": 32,
#         "seed": 41
#     }
#
#     from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld
#     from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
#
#     LAYOUT = "sanity_check2"; HORIZON = 500; ITERATIONS = 10_000
#
#     # Generate MDP and environment----------------
#     base_mdp = OvercookedGridworld.from_layout_name(LAYOUT)
#     env = OvercookedEnv.from_mdp(base_mdp, horizon=HORIZON)
#     # reply_memory = ReplayMemory(capacity=1000)
#     # print(f'Layout: {LAYOUT} | Shape: {base_mdp.shape} | Iterations: {ITERATIONS}'')
#
#     # Generate agents
#     # model = DQN(**model_config)
#     q_agent = SoloDeepQAgent(base_mdp,agent_index=0,model=DQN,config=model_config)
#     stay_agent = StayAgent()
#     # agents = [CustomQAgent(mdp, is_learning_agent=True, save_agent_file=True), StayAgent()]
#
#     # base_env.reset()
#     # state = base_env.state
#     # action = Action.ALL_ACTIONS[0]
#     # reward = 1
#     # next_state = base_env.state
#     # explore_decay_prog = 0.5
#     # q_agent.update(state, action, reward, next_state, explore_decay_prog)
#
#
#     for test in range(n_tests):
#         env.reset()
#         cum_reward = 0
#         for t in count():
#             state = env.state
#             # q_agent.model.predict(q_agent.featurize(state)[np.newaxis])
#             action1, _ = q_agent.action(state)
#             action2, _ = stay_agent.action(state)
#             joint_action = (action1, action2)
#             next_state, reward, done, info = env.step(joint_action)  # what is joint-action info?
#             reward += info["shaped_r_by_agent"][0]
#             # if  info["shaped_r_by_agent"][0] != 0:
#             #     print(f"Shaped reward {info['shaped_r_by_agent'][0]}")
#             # q_agent.update(state, action1, reward, next_state, explore_decay_prog=total_updates / (HORIZON * ITERATIONS))
#             # cum_reward += reward
#
#
#             # reply_memory.push(q_agent.featurize(state), Action.ACTION_TO_INDEX[action1], reward, q_agent.featurize(next_state), done)
#             replay_memory.append((q_agent.featurize(state), Action.ACTION_TO_INDEX[action1], reward, q_agent.featurize(next_state), done))
#             # print(base_mdp.get_recipe_value(state))
#             for obj in state.all_objects_list:
#                 if 'soup' ==obj.name:
#                     if obj.is_cooking:
#                         assert len(obj.ingredients)==3, 'Cooking soup should have 3 ingredients'
#                         print('Cooking soup')
#                     if obj.is_ready:
#                         print('Soup is ready')
#                     # print(f'Soup: {obj.ingredients} | is cooking={obj.is_cooking} | is ready={obj.is_ready} | is idle={obj.is_idle}')
#
#             verify_lossless_featurization(base_mdp, q_agent, state)
#             if done:
#                 break
#         # print(f'Obs shape:{np.shape(q_agent.featurize(state))}')
#
#         # experineces = reply_memory.sample(model_config['minibatch_size'])
#         experineces = sample_experiences(model_config['minibatch_size'])
#         q_agent.update(experineces)
#
#         print(f"Test {test} passed")

# DEPRECATED ########################################################
# class SoloDeepQAgent(Agent):
#     """An agent randomly picks motion actions.
#     Note: Does not perform interat actions, unless specified"""
#
#     def __init__(self, mdp, agent_index, policy_net, # model,
#                  save_agent_file=False, load_learned_agent=False,
#                  config=None,verbose_load_config=True):
#         super(SoloDeepQAgent, self).__init__()
#
#         # Learning Params ---------------
#         self.gamma = 0.95
#         self.lr = 0.05
#         self.load_learned_agent = load_learned_agent
#         self.save_agent_file = save_agent_file
#         self.tabular_dtype = np.float32
#         self.horizon = 400
#
#
#         # Featurize Params ---------------
#         self.my_index = agent_index
#         self.partner_index = int(not self.my_index)
#         self.IDX_TO_OBJ = ["onion", "dish", "soup"]
#         self.OBJ_TO_IDX = {o_name: idx for idx, o_name in enumerate(self.IDX_TO_OBJ)}
#         self.reachable_counters = mdp.get_reachable_counters()
#         self.featurize_dtype = np.int32
#         self.featurize_fn = config['featurize_fn']
#
#         # Create learning environment ---------------
#         self.mdp = mdp
#         # self.valid_actions = Action.ALL_ACTIONS
#         self.action_space = Action.ALL_ACTIONS
#
#         config['obs_shape'] =self.get_featurized_shape()
#
#         if config is not None: self.load_config(config,verbose=verbose_load_config)
#         else: warnings.warn("No config provided, using default values. Please ensure defaults are correct")
#
#         self.policy_net = policy_net
#         self.device = config['device']
#
#         # self.model = model(**config)  # DQN model used for learning
#
#     def load_config(self, config,verbose=True):
#         if verbose:
#             print(f'--------------------------------------------------')
#             print(f"Loading config for agent {self.my_index}:")
#             print(f'--------------------------------------------------')
#         for key, value in config.items():
#             setattr(self, key, value)
#             if verbose:  print(f"\t|{key}: {value}")
#         if verbose:
#             print(f'--------------------------------------------------')
#     def print_config(self):
#         report_list = ['exploration_proba', 'max_exploration_proba', 'min_exploration_proba', 'gamma', 'lr', 'rationality']
#         print(f'--------------------------------------------------')
#         print(f'Agent {self.my_index} Config:')
#         print(f'--------------------------------------------------')
#         for key in report_list:
#             print(f"\t|{key}: {getattr(self, key)}")
#         print(f'--------------------------------------------------\n')
#     ########################################################
#     # Learning/Performance methods #########################
#     ########################################################
#     def action(self, state, exp_prob=0, rationality='max',debug=False):
#     # def action(self, state,enable_explore=True,rationality=None):
#         # DYNAMIC EXPLORATION E-GREEDY/BOLTZMAN
#         # Epsilon greedy ----------------
#
#         # if rationality is None: rationality = self.rationality
#         #
#         # if rationality == 'max':
#         #     obs = self.featurize(state)
#         #     # qs = self.model.predict(obs[np.newaxis]).flatten()
#         #     qs = self.model.predict(obs[np.newaxis]).flatten()
#         #     action_idx = np.argmax(qs)
#         #     # action = self.valid_actions[action_idx]
#         #     action = Action.INDEX_TO_ACTION[action_idx]
#         #     action_probs = np.eye(Action.NUM_ACTIONS)[action_idx]
#         #     action_info = {"action_probs": action_probs}
#         # elif rationality == 'random' or rationality < 0:
#         #     action_probs = np.ones(len(self.valid_actions)) / len(self.valid_actions)
#         #     action = Action.sample(action_probs)
#         #     action_info = {"action_probs": action_probs}
#         # else:
#         #     obs = self.featurize(state)
#         #     qs = self.model.predict(obs[np.newaxis]).flatten()
#         #     # action_probs = np.exp(rationality * qs) / np.sum(np.exp(rationality * qs))
#         #     action_probs = self.softmax(self.rationality * qs)
#         #     action = Action.sample(action_probs)
#         #     action_info = {"action_probs": action_probs}
#         # return action, action_info
#     # def action(self, state, exp_prob,debug=False,rationality=None):
#         """
#         :param state: OvercookedState object
#         :param exp_prob: probability of random exploration {0: always exploit, 1: always explore}
#         :param rationality: if not exploring, Boltzmann rationality temperature ('max' if argmax)
#         :return: action, action_info = {"action_index": int, "action_probs": np.ndarray}
#         """
#         # global steps_done
#         sample = random.random()
#
#         # EXPLORE ----------------
#         if sample < exp_prob:
#             action_probs = np.ones(len(self.action_space)) / len(self.action_space)
#             ai = np.random.choice(np.arange(len(self.action_space)), p=action_probs)
#             action = self.action_space[ai]
#
#         # EXPLOIT ----------------
#         else:
#             # obs = self.mdp.get_lossless_encoding_vector(state)
#             # obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
#             obs = self.featurize(state)
#             with torch.no_grad():
#                 if rationality == 'max':
#                     ai = self.policy_net(obs).max(1).indices.view(1, 1).numpy().flatten()[0]
#                     action_probs = np.eye(len(self.action_space))[ai]
#                     action = self.action_space[ai]
#                 else:
#                     qA = self.policy_net(obs).numpy().flatten()
#                     action_probs = self.softmax(rationality * qA)
#                     ai = np.random.choice(self.action_space, p=action_probs)
#                     action = self.action_space[ai]
#
#                     # return torch.tensor([[ai]], device=device, dtype=torch.long)
#         # RETURN ----------------
#         action_info = {"action_index": ai, "action_probs": action_probs}
#         return action, action_info
#
#         # if rationality is not None:
#         #     if rationality == 'max':
#         #         with torch.no_grad():
#         #             # t.max(1) will return the largest column value of each row.
#         #             # second column on max result is index of where max element was
#         #             # found, so we pick action with the larger expected reward.
#         #             # return policy_net(state).max(1).indices.view(1, 1).numpy().flatten()[0]
#         #             return policy_net(state).max(1).indices.view(1, 1)
#         #     else:
#         #         with torch.no_grad():
#         #             qA = policy_net(state).numpy().flatten()
#         #             ex = np.exp(rationality * (qA - np.max(qA)))
#         #             pA = ex / np.sum(ex)
#         #             action = Action.sample(pA)
#         #             ai = Action.ACTION_TO_INDEX[action]
#         #             return torch.tensor([[ai]], device=device, dtype=torch.long)
#         # elif sample < exp_prob:
#         #     # return np.random.choice(np.arange(n_actions))
#         #     action = Action.sample(np.ones(n_actions) / n_actions)
#         #     ai = Action.ACTION_TO_INDEX[action]
#         #     return torch.tensor([[ai]], device=device, dtype=torch.long)
#         #     # return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
#         # else:
#         #     if debug: print('Greedy')
#         #     with torch.no_grad():
#         #         # t.max(1) will return the largest column value of each row.
#         #         # second column on max result is index of where max element was
#         #         # found, so we pick action with the larger expected reward.
#         #         # return policy_net(state).max(1).indices.view(1, 1).numpy().flatten()[0]
#         #         return policy_net(state).max(1).indices.view(1, 1)
#
#     def actions(self, states, agent_indices):
#         return (self.action(state) for state in states)
#
#     def softmax(self, x):
#         e_x = np.exp(x- np.max(x))
#         # e_x = np.exp(x)
#         return e_x / e_x.sum()
#
#     def update(self, experiences):
#         """ Provides batch update to the DQN model """
#         states, actions, rewards, next_states, dones = experiences
#         next_Q_values =   self.model.predict_batch_nograd(next_states) #self.model.predict(next_states)
#         max_next_Q_values = np.max(next_Q_values, axis=1)
#         target_Q_values = (rewards + (1 - dones) * self.gamma * max_next_Q_values)
#         target_Q_values = target_Q_values.reshape(-1, 1)
#         target_Q_values = self.valuation_fun(target_Q_values)  # apply possibly biased valuation function
#         mask = tf.one_hot(actions, self.model.n_outputs)
#         with tf.GradientTape() as tape:
#             all_Q_values = self.model.base_model(states) # must call model directly for gradient calc
#             Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
#             loss = tf.reduce_mean(self.model.loss_fn(target_Q_values, Q_values))
#         grads = tape.gradient(loss, self.model.trainable_variables)
#         self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
#
#     def valuation_fun(self, td_target):
#         """ Applies CPT (if specified) to the target Q values"""
#         return td_target
#
#     ########################################################
#     # Featurization methods ################################
#     ########################################################
#     def featurize(self, overcooked_state):
#         """ Generate either a feature vector or series of masks for CNN"""
#         if self.featurize_fn.lower() in ['mask', 'lossless']:
#             return self.featurize_mask(overcooked_state)
#         elif self.featurize_fn.lower() in ['handcraft_vector','vector']:
#             # return self.featurize_vector(overcooked_state)
#             obs = self.mdp.get_lossless_encoding_vector(overcooked_state)
#             obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
#             return obs
#
#     def get_featurized_shape(self):
#         """ Get the shape of either type of the featurized state"""
#         if self.featurize_fn.lower() in ['mask', 'lossless']:
#             return np.shape(self.featurize_mask(self.mdp.get_standard_start_state()))
#         elif self.featurize_fn.lower() in ['handcraft_vector']:
#             # return self.get_featurized_vector_shape()
#             return self.mdp.get_lossless_encoding_vector_shape()
#
#     # Mask Featurization ###############################
#     def featurize_mask(self, overcooked_state, get_keys=False):
#         ordered_player_features = self.ordered_player_mask(overcooked_state)
#         base_map_features = self.base_map_mask()
#         variable_map_features = self.variable_map_mask(overcooked_state)
#         # urgency_map_features = self.urgency_map_mask(overcooked_state)
#         feature_masks = np.array(
#             list(ordered_player_features.values()) +\
#             list(base_map_features.values()) +\
#             list(variable_map_features.values()) #+\
#             # list(urgency_map_features.values())
#         )
#         feature_mask_keys = tuple(
#             list(ordered_player_features.keys()) + \
#             list(base_map_features.keys()) + \
#             list(variable_map_features.keys())  # + \
#             # list(urgency_map_features.keys())
#         )
#         # !! IF USING PYTORCH !!
#         assert feature_masks.shape[1:] == self.mdp.shape
#         assert feature_masks.shape[0] == len(feature_mask_keys)
#         # !!! if using tensorflow!!!!
#         # feature_masks = np.transpose(feature_masks, (1, 2, 0))
#         # assert feature_masks.shape[:2] == self.mdp.shape
#         # assert feature_masks.shape[2] == len(feature_mask_keys)
#
#         if get_keys:
#             return feature_masks,feature_mask_keys
#         return feature_masks
#
#     def ordered_player_mask(self, state):
#         """
#         #TODO: Make it so you do not have to recalculate during self-play
#         :param state: OvercookedState object
#         :return: features: ordered_player_features
#         """
#         features = {
#             'player_0_loc': np.zeros(self.mdp.shape,dtype=np.int),
#             # 'player_1_loc': np.zeros(self.mdp.shape,dtype=np.int),
#             'player_0_orientation_0': np.zeros(self.mdp.shape,dtype=np.int),
#             'player_0_orientation_1': np.zeros(self.mdp.shape,dtype=np.int),
#             'player_0_orientation_2': np.zeros(self.mdp.shape,dtype=np.int),
#             'player_0_orientation_3': np.zeros(self.mdp.shape,dtype=np.int),
#             # 'player_1_orientation_0': np.zeros(self.mdp.shape,dtype=np.int),
#             # 'player_1_orientation_1': np.zeros(self.mdp.shape,dtype=np.int),
#             # 'player_1_orientation_2': np.zeros(self.mdp.shape,dtype=np.int),
#             # 'player_1_orientation_3': np.zeros(self.mdp.shape,dtype=np.int)
#         }
#         # Solo player observation ----------------
#         player_idx = self.my_index
#         player = state.players[player_idx]
#         features[f'player_{player_idx}_loc'][player.position] = 1
#         features[f'player_{player_idx}_orientation_{Direction.DIRECTION_TO_INDEX[player.orientation]}'][player.position] = 1
#
#         # Joint player observation ----------------
#         # for player_idx, player in enumerate(state.players):
#         #     features[f'player_{player_idx}_loc'][player.position] = 1
#         #     features[f'player_{player_idx}_orientation_{Direction.DIRECTION_TO_INDEX[player.orientation]}'][player.position] = 1
#
#
#         return features
#
#     def base_map_mask(self):
#         features = {
#             "pot_loc":          np.zeros(self.mdp.shape,dtype=np.int),
#             "counter_loc":      np.zeros(self.mdp.shape,dtype=np.int),
#             "onion_disp_loc":   np.zeros(self.mdp.shape,dtype=np.int),
#             # "tomato_disp_loc":np.zeros(self.mdp.shape,dtype=np.int),
#             "dish_disp_loc":    np.zeros(self.mdp.shape,dtype=np.int),
#             "serve_loc":        np.zeros(self.mdp.shape,dtype=np.int),
#             "water_loc":        np.zeros(self.mdp.shape,dtype=np.int),
#         }
#
#         for loc in self.mdp.get_pot_locations():                features["pot_loc"][loc] = 1
#         for loc in self.mdp.get_counter_locations():            features["counter_loc"][loc] = 1
#         for loc in self.mdp.get_onion_dispenser_locations():    features["onion_disp_loc"][loc] = 1
#         # for loc in self.mdp.get_tomato_dispenser_locations():   features["tomato_disp_loc"][loc] = 1
#         for loc in self.mdp.get_dish_dispenser_locations():     features["dish_disp_loc"][loc] = 1
#         for loc in self.mdp.get_serving_locations():            features["serve_loc"][loc] = 1
#         for loc in self.mdp.get_water_locations():              features["water_loc"][loc] = 1
#         return features
#
#     def variable_map_mask(self, state):
#         features = {
#             "onions_in_pot": np.zeros(self.mdp.shape,dtype=np.int),
#             # "tomatoes_in_pot": np.zeros(self.mdp.shape,dtype=np.int),
#             "onions_in_soup": np.zeros(self.mdp.shape,dtype=np.int),            # val = [0,1,2,3]
#             # "tomatoes_in_soup": np.zeros(self.mdp.shape,dtype=np.int),
#             "soup_cook_time_remaining": np.zeros(self.mdp.shape,dtype=np.int),  # val = remaining game ticks
#             "soup_done": np.zeros(self.mdp.shape,dtype=np.int),                 # val = bool
#             "dishes": np.zeros(self.mdp.shape,dtype=np.int),
#             "onions": np.zeros(self.mdp.shape,dtype=np.int),
#             # "tomatoes": np.zeros(self.mdp.shape),
#         }
#         for obj in state.all_objects_list:
#             if obj.name == "soup":
#                 # get the ingredients into a {object: number} dictionary
#                 ingredients_dict = Counter(obj.ingredients)
#                 if obj.position in self.mdp.get_pot_locations():
#                     if obj.is_idle:
#                         # onions_in_pot and tomatoes_in_pot are used when the soup is idling, and ingredients could still be added
#                         features["onions_in_pot"][obj.position] += ingredients_dict["onion"]
#                         # features["tomatoes_in_pot"][obj.position] += ingredients_dict["tomato"]
#                     else:
#                         features["onions_in_soup"][obj.position] += ingredients_dict["onion"]
#                         # features["tomatoes_in_soup"][obj.position] +=  ingredients_dict["tomato"]
#                         features["soup_cook_time_remaining" ][obj.position] += obj.cook_time - obj._cooking_tick
#                         if obj.is_ready:  features["soup_done"][obj.position] += 1
#                 else:
#                     # If player soup is not in a pot, treat it like a soup that is cooked with remaining time 0
#                     features["onions_in_soup"][obj.position] += ingredients_dict["onion"]
#                     # features["tomatoes_in_soup"][obj.position] +=  ingredients_dict["tomato"]
#                     features["soup_done"][obj.position] +=  1
#
#             elif obj.name == "dish": features["dishes"][obj.position] += 1
#             elif obj.name == "onion": features["onions"][obj.position] += 1
#             # elif obj.name == "tomato": features["tomatoes"][obj.position] += 1
#             else:  raise ValueError("Unrecognized object")
#         return features
#
#     def urgency_map_mask(self, state):
#         features = {
#             "urgency": np.zeros(self.mdp.shape,dtype=np.int)
#         }
#         if self.horizon - state.timestep < 40: features["urgency"] = np.ones(self.mdp.shape)
#         return features
#
#     # Vector Featurization #############################
#     def featurize_vector(self,overcooked_state):
#         """
#         Takes in OvercookedState object and converts to observation index conducive to learning
#         :param overcooked_state: OvercookedState object
#         :return: observation feature vector for ego player
#             ego_features: len = 2+4+3 = 9
#                 - pi_position: length 2 list of x,y position of player i
#                 - pi_orientation: length 4 one-hot-encoding of direction currently facing
#                 - pi_obj: length n_asset (3=|onion,plate,soup|) one-hot-encoding of object currently holding
#             partner_features: len = 2+4+3 = 9
#                 - pj_position: length 2 list of x,y position of player i
#                 - pj_orientation: length 4 one-hot-encoding of direction currently facing
#                 - pj_obj: length n_asset (3=|onion,plate,soup|) one-hot-encoding of object currently holding
#                 TODO: Reduce feature space by removing orientation from partner player (pj)?
#             world_features: n_counter+n_pot
#                 - counter_status: length n_counter of asset labels on counters {'nothing': 0, 'onion': 1, 'dish': 2, 'soup': 3}
#                 TODO: make 1-hot encoding of counter status/decouple corralation between counter and object?
#                 - pot_status: length n_pot of asset labels in pots {'0 onion': 0, '1 onion': 1, '2 onion': 2, '3 onion': 3, 'ready': 4}
#
#             OTHER POSSIBLE FEATURES:
#                 - distance each static asset: d_asset = [dx,dy] (len = n_asset x 2) to contextualize goals
#                     Assets: pot, ingrediant source, serving counter, trash, puddle/water
#         """
#         # ego_features = self.featurize_player(overcooked_state, self.my_index)
#         # partner_features = self.featurize_player(overcooked_state, self.partner_index)
#         # world_features = self.featurize_world(overcooked_state)
#         # return np.concatenate([ego_features, partner_features, world_features]).astype(self.featurize_dtype)
#         ego_features = self.featurize_player_vector(overcooked_state, self.my_index)
#         world_features = self.featurize_world_vector(overcooked_state)
#         return np.concatenate([ego_features, world_features]).astype(self.featurize_dtype)
#
#     def featurize_player_vector(self, state, i):
#         """
#         Features:
#             - pi_position: length 2 list of x,y position of player i
#             - pi_orientation: length 4 one-hot-encoding of direction currently facing
#             - pi_holding: length n_asset (3=|onion,plate,soup|) one-hot-encoding of object currently holding
#         :param state: OvecookedState object
#         :param i: which player to get features for
#         :return: player_feature_vector
#         """
#         player_features = {}
#         player = state.players[i]
#
#         # Get position features ---------------
#         player_features["p{}_position".format(i)] = np.array(player.position)
#
#         # Get orientation features ---------------
#         orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
#         player_features["p{}_orientation".format(i)] = np.eye(4)[orientation_idx]
#
#         # Get holding features (1-HOT)---------------
#         obj = player.held_object
#         if obj is None:
#             held_obj_name = "none"
#             player_features["p{}_objs".format(i)] = np.zeros(len(self.IDX_TO_OBJ))
#         else:
#             held_obj_name = obj.name
#             obj_idx = self.OBJ_TO_IDX[held_obj_name]
#             player_features["p{}_objs".format(i)] = np.eye(len(self.IDX_TO_OBJ))[obj_idx]
#
#         # Create feature vector ---------------
#         player_feature_vector = np.concatenate([player_features["p{}_position".format(i)],
#                                                 player_features["p{}_orientation".format(i)],
#                                                 player_features["p{}_objs".format(i)]])
#
#
#         return player_feature_vector
#
#     def featurize_world_vector(self, overcooked_state):
#         """
#         TODO: validate difference between full_but_not_cooking_pots and cooking_pots; doesnt it start automatically?
#         Features:
#             - counter_status: length n_counter of asset labels on counters {'nothing': 0, 'onion': 1, 'dish': 2, 'soup': 3}
#             TODO: make 1-hot encoding of counter status/decouple corralation between counter and object?
#             - pot_status: length n_pot of asset labels in pots
#                 {'empty': 0, '1 onion': 1, '2 onion': 2, '3 onion/cooking': 3, 'ready': 4}
#                 {'empty': 0, 'X items': X,..., 'cooking': 3, 'ready': 4}
#             TODO: remove number of items in pot and instead just label as {'empty': 0, 'not_full':1, 'full':2, 'cooking':3, 'ready':4}
#
#         # Other pot state info cmds
#         # pot_states = self.mdp.get_pot_states(overcooked_state)
#         # is_empty = int(pot_loc in self.mdp.get_empty_pots(pot_states))
#         # is_full = int(pot_loc in self.mdp.get_full_but_not_cooking_pots(pot_states))
#         # is_cooking = int(pot_loc in self.mdp.get_cooking_pots(pot_states))
#         # is_ready = int(pot_loc in self.mdp.get_ready_pots(pot_states))
#         # is_partially_ful = int(pot_loc in self.mdp.get_partially_full_pots(pot_states))
#
#         :param overcooked_state: OvercookedState object
#         :return: world_feature_vector
#         """
#         world_features = {}
#
#         # get counter status feature vector (LABELED) ---------------
#         # counter_locs = self.reachable_counters # self.mdp.get_counter_locations()
#         # counter_labels = np.zeros(len(counter_locs))
#         # counter_objs = self.mdp.get_counter_objects_dict(overcooked_state) # dictionary of pos:objects
#         # for counter_loc, counter_obj in counter_objs.items():
#         #     counter_labels[counter_locs.index(counter_loc)] = self.OBJ_TO_IDX[counter_obj]
#         # world_features["counter_status"] = counter_labels
#
#         # get counter status feature vector (1-Hot) ---------------
#         counter_locs = self.reachable_counters  # self.mdp.get_counter_locations()
#         counter_indicator_arr = np.zeros([len(counter_locs), len(self.IDX_TO_OBJ)])
#         counter_objs = self.mdp.get_counter_objects_dict(overcooked_state)  # dictionary of pos:objects
#         for counter_obj,counter_loc in counter_objs.items():
#             iobj = self.OBJ_TO_IDX[counter_obj]
#             icounter = counter_locs.index(counter_loc[0])
#             counter_indicator_arr[icounter,iobj] = 1
#         world_features["counter_status"] = counter_indicator_arr.flatten()
#
#         # get pot status feature vector ---------------
#         req_ingredients = self.mdp.recipe_config['num_items_for_soup'] # number of ingrediants before cooking
#         pot_locs = self.mdp.get_pot_locations()
#         pot_labels = np.zeros(len(pot_locs))
#         for pot_index, pot_loc in enumerate(pot_locs):
#             is_empty = not overcooked_state.has_object(pot_loc)
#             if is_empty: pot_labels[pot_index] = 0
#             else:
#                 soup = overcooked_state.get_object(pot_loc)
#                 if soup.is_ready:               pot_labels[pot_index]= req_ingredients + 1
#                 elif soup.is_cooking:           pot_labels[pot_index]= req_ingredients
#                 elif len(soup.ingredients) >0:  pot_labels[pot_index] = len(soup.ingredients)
#                 else: raise ValueError(f"Invalid pot state {soup}")
#
#
#         world_features["pot_status"] = pot_labels
#
#         # Create feature vector ---------------
#         world_feature_vector = np.concatenate([world_features['counter_status'],
#                                                world_features['pot_status']])
#         return world_feature_vector
#
#     def get_featurized_vector_shape(self):
#         n_features = len(self.featurize(self.mdp.get_standard_start_state()))
#         d_1hot = 2
#
#         # Player features
#         pos_dim = list(np.shape(self.mdp.terrain_mtx))[::-1]
#         orientation_dim = [d_1hot for _ in range(len(Direction.ALL_DIRECTIONS))]
#         hold_dim = [d_1hot for _ in range(len(self.IDX_TO_OBJ))]
#         player_dim = pos_dim + orientation_dim + hold_dim
#
#         # World features
#         counter_dim = [d_1hot for _ in range(len(self.reachable_counters)* len(self.IDX_TO_OBJ))]
#         pot_dim = [(self.mdp.recipe_config['num_items_for_soup']+2) * len(self.mdp.get_pot_locations())]
#         world_dim = counter_dim + pot_dim
#
#         # Full feature dim
#         # feature_dim = player_dim + player_dim + world_dim
#         feature_dim = player_dim + world_dim
#         assert len(feature_dim) == n_features, f"Feature dim {len(feature_dim)} != n_features {n_features}"
#         # return feature_dim
#
#         return np.shape(feature_dim)
#
#     ########################################################
#     # Saving utils #########################################
#     ########################################################
#     def save_agent(self, filename):
#         """
#         Save the Q-table to a file.
#
#         Args:
#             filename (str): The name of the file to save the Q-table to.
#
#         Returns:
#             None
#         """
#         # filename = os.path.join(os.path.dirname(__file__), filename)
#         filename = os.path.join(self.learned_agent_path, filename)
#         with open(filename, 'wb') as f:
#             pickle.dump(self.Q_table, f)
#
#     def load_agent(self, filename):
#         """
#         Load the Q-table from a file.
#
#         Args:
#             filename (str): The name of the file to load the Q-table from.
#
#         Returns:
#             None
#         """
#         filename = os.path.join(os.path.dirname(__file__), filename)
#         with open(filename, 'rb') as f:
#             self.q_table = pickle.load(f)
