import warnings

import numpy as np
from risky_overcooked_py.mdp.actions import Action, Direction
from risky_overcooked_py.agents.agent import Agent, AgentPair,StayAgent, RandomAgent, GreedyHumanModel
import pickle
import datetime
import os
from risky_overcooked_rl.utils.deep_models_tf import DQN,ReplayMemory
from collections import Counter, defaultdict
import tensorflow as tf
from itertools import product,count
import torch
from collections import namedtuple, deque
replay_memory = deque(maxlen=20000)
def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_memory), size=batch_size)
    batch = [replay_memory[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones

class SoloDeepQAgent(Agent):
    """An agent randomly picks motion actions.
    Note: Does not perform interat actions, unless specified"""

    def __init__(self, mdp, agent_index, model,
                 save_agent_file=False, load_learned_agent=False,
                 config=None,verbose_load_config=True,featurize_fn='mask'):
        super(SoloDeepQAgent, self).__init__()

        # Learning Params ---------------
        # self.max_exploration_proba = 0.9
        # self.min_exploration_proba = 0.01
        # self.exploration_proba = self.max_exploration_proba
        self.gamma = 0.95
        self.lr = 0.05
        self.rationality = 4.0
        self.load_learned_agent = load_learned_agent
        self.save_agent_file = save_agent_file
        self.tabular_dtype = np.float32
        self.horizon = 400
        self.device = config['device']


        # Featurize Params ---------------
        self.my_index = agent_index
        self.partner_index = int(not self.my_index)
        self.IDX_TO_OBJ = ["onion", "dish", "soup"]
        self.OBJ_TO_IDX = {o_name: idx for idx, o_name in enumerate(self.IDX_TO_OBJ)}
        self.reachable_counters = mdp.get_reachable_counters()
        self.featurize_dtype = np.int32
        self.featurize_fn = featurize_fn

        # Create learning environment ---------------
        self.mdp = mdp
        self.valid_actions = Action.ALL_ACTIONS

        config['obs_shape'] =self.get_featurized_shape()

        if config is not None: self.load_config(config,verbose=verbose_load_config)
        else: warnings.warn("No config provided, using default values. Please ensure defaults are correct")

        self.model = model(**config)  # DQN model used for learning

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

    def action(self, state,enable_explore=True,rationality=None):
        # TODO: player position is not enough to determine quality of action. What about if holding item, orientation/facing, ect...
        # DYNAMIC EXPLORATION E-GREEDY/BOLTZMAN
        # Epsilon greedy ----------------
        # if np.random.uniform(0, 1) < self.exploration_proba and enable_explore:
        #     action_probs = np.zeros(Action.NUM_ACTIONS)
        #     legal_actions = Action.ALL_ACTIONS
        #     legal_actions_indices = np.array([Action.ACTION_TO_INDEX[motion_a] for motion_a in legal_actions])
        #     action_probs[legal_actions_indices] = 1 / len(legal_actions)
        #
        #     action = Action.sample(action_probs)
        #     action_info = {"action_probs": action_probs}
        #
        # else:
        #     # obs = state # for debugging
        #     obs = self.featurize(state)
        #     qs = self.model.predict(obs[np.newaxis]).flatten()
        #     # Boltzman Agent -----------
        #     if self.rationality == 'max':
        #         action_idx = np.argmax(qs)
        #         action = self.valid_actions[action_idx]
        #         action_probs = np.eye(Action.NUM_ACTIONS)[action_idx]
        #         action_info = {"action_probs": action_probs}
        #     else:
        #         # action_probs = np.exp(rationality * qs) / np.sum(np.exp(rationality * qs))
        #         action_probs = self.softmax(self.rationality * qs)
        #         action = Action.sample(action_probs)
        #         action_info = {"action_probs": action_probs}

        if rationality is None: rationality = self.rationality



        if rationality == 'max':
            obs = self.featurize(state)
            # qs = self.model.predict(obs[np.newaxis]).flatten()
            qs = self.model.predict(obs[np.newaxis]).flatten()
            action_idx = np.argmax(qs)
            # action = self.valid_actions[action_idx]
            action = Action.INDEX_TO_ACTION[action_idx]
            action_probs = np.eye(Action.NUM_ACTIONS)[action_idx]
            action_info = {"action_probs": action_probs}
        elif rationality == 'random' or rationality < 0:
            # action_probs = np.ones(len(self.valid_actions)) / len(self.valid_actions)
            # action = Action.sample(action_probs)
            # action_info = {"action_probs": action_probs}
            # return np.random.choice(np.arange(n_actions))
            action = Action.sample(np.ones(len(self.valid_actions)) / len(self.valid_actions))
            ai = Action.ACTION_TO_INDEX[action]
            return torch.tensor([[ai]], device=self.device, dtype=torch.long)
        else:
            obs = self.featurize(state)
            qs = self.model.predict(obs[np.newaxis]).flatten()
            # action_probs = np.exp(rationality * qs) / np.sum(np.exp(rationality * qs))
            action_probs = self.softmax(self.rationality * qs)
            action = Action.sample(action_probs)
            action_info = {"action_probs": action_probs}
        return action, action_info

    def actions(self, states, agent_indices):
        return (self.action(state) for state in states)

    def softmax(self, x):
        e_x = np.exp(x- np.max(x))
        # e_x = np.exp(x)
        return e_x / e_x.sum()

    def update(self, experiences):
        """ Provides batch update to the DQN model """
        states, actions, rewards, next_states, dones = experiences
        next_Q_values =   self.model.predict_batch_nograd(next_states) #self.model.predict(next_states)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards + (1 - dones) * self.gamma * max_next_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1)
        target_Q_values = self.valuation_fun(target_Q_values)  # apply possibly biased valuation function
        mask = tf.one_hot(actions, self.model.n_outputs)
        with tf.GradientTape() as tape:
            all_Q_values = self.model.base_model(states) # must call model directly for gradient calc
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.model.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def valuation_fun(self, td_target):
        """ Applies CPT (if specified) to the target Q values"""
        return td_target

    ########################################################
    # Featurization methods ################################
    ########################################################
    def featurize(self, overcooked_state):
        """ Generate either a feature vector or series of masks for CNN"""
        if self.featurize_fn.lower() in ['mask', 'lossless']:
            return self.featurize_mask(overcooked_state)
        elif self.featurize_fn.lower() in ['handcraft_vector']:
            return self.featurize_vector(overcooked_state)

    def get_featurized_shape(self):
        """ Get the shape of either type of the featurized state"""
        if self.featurize_fn.lower() in ['mask', 'lossless']:
            return np.shape(self.featurize_mask(self.mdp.get_standard_start_state()))
        elif self.featurize_fn.lower() in ['handcraft_vector']:
            return self.get_featurized_vector_shape()

    # Mask Featurization ###############################
    def featurize_mask(self, overcooked_state, get_keys=False):
        ordered_player_features = self.ordered_player_mask(overcooked_state)
        base_map_features = self.base_map_mask()
        variable_map_features = self.variable_map_mask(overcooked_state)
        # urgency_map_features = self.urgency_map_mask(overcooked_state)
        feature_masks = np.array(
            list(ordered_player_features.values()) +\
            list(base_map_features.values()) +\
            list(variable_map_features.values()) #+\
            # list(urgency_map_features.values())
        )
        feature_mask_keys = tuple(
            list(ordered_player_features.keys()) + \
            list(base_map_features.keys()) + \
            list(variable_map_features.keys())  # + \
            # list(urgency_map_features.keys())
        )

        feature_masks = np.transpose(feature_masks, (1, 2, 0))
        assert feature_masks.shape[:2] == self.mdp.shape
        assert feature_masks.shape[2] == len(feature_mask_keys)

        if get_keys:
            return feature_masks,feature_mask_keys
        return feature_masks

    def ordered_player_mask(self, state):
        """
        #TODO: Make it so you do not have to recalculate during self-play
        :param state: OvercookedState object
        :return: features: ordered_player_features
        """
        features = {
            'player_0_loc': np.zeros(self.mdp.shape,dtype=np.int),
            # 'player_1_loc': np.zeros(self.mdp.shape,dtype=np.int),
            'player_0_orientation_0': np.zeros(self.mdp.shape,dtype=np.int),
            'player_0_orientation_1': np.zeros(self.mdp.shape,dtype=np.int),
            'player_0_orientation_2': np.zeros(self.mdp.shape,dtype=np.int),
            'player_0_orientation_3': np.zeros(self.mdp.shape,dtype=np.int),
            # 'player_1_orientation_0': np.zeros(self.mdp.shape,dtype=np.int),
            # 'player_1_orientation_1': np.zeros(self.mdp.shape,dtype=np.int),
            # 'player_1_orientation_2': np.zeros(self.mdp.shape,dtype=np.int),
            # 'player_1_orientation_3': np.zeros(self.mdp.shape,dtype=np.int)
        }
        # Solo player observation ----------------
        player_idx = self.my_index
        player = state.players[player_idx]
        features[f'player_{player_idx}_loc'][player.position] = 1
        features[f'player_{player_idx}_orientation_{Direction.DIRECTION_TO_INDEX[player.orientation]}'][player.position] = 1

        # Joint player observation ----------------
        # for player_idx, player in enumerate(state.players):
        #     features[f'player_{player_idx}_loc'][player.position] = 1
        #     features[f'player_{player_idx}_orientation_{Direction.DIRECTION_TO_INDEX[player.orientation]}'][player.position] = 1


        return features

    def base_map_mask(self):
        features = {
            "pot_loc":          np.zeros(self.mdp.shape,dtype=np.int),
            "counter_loc":      np.zeros(self.mdp.shape,dtype=np.int),
            "onion_disp_loc":   np.zeros(self.mdp.shape,dtype=np.int),
            # "tomato_disp_loc":np.zeros(self.mdp.shape,dtype=np.int),
            "dish_disp_loc":    np.zeros(self.mdp.shape,dtype=np.int),
            "serve_loc":        np.zeros(self.mdp.shape,dtype=np.int),
            "water_loc":        np.zeros(self.mdp.shape,dtype=np.int),
        }

        for loc in self.mdp.get_pot_locations():                features["pot_loc"][loc] = 1
        for loc in self.mdp.get_counter_locations():            features["counter_loc"][loc] = 1
        for loc in self.mdp.get_onion_dispenser_locations():    features["onion_disp_loc"][loc] = 1
        # for loc in self.mdp.get_tomato_dispenser_locations():   features["tomato_disp_loc"][loc] = 1
        for loc in self.mdp.get_dish_dispenser_locations():     features["dish_disp_loc"][loc] = 1
        for loc in self.mdp.get_serving_locations():            features["serve_loc"][loc] = 1
        for loc in self.mdp.get_water_locations():              features["water_loc"][loc] = 1
        return features

    def variable_map_mask(self, state):
        features = {
            "onions_in_pot": np.zeros(self.mdp.shape,dtype=np.int),
            # "tomatoes_in_pot": np.zeros(self.mdp.shape,dtype=np.int),
            "onions_in_soup": np.zeros(self.mdp.shape,dtype=np.int),            # val = [0,1,2,3]
            # "tomatoes_in_soup": np.zeros(self.mdp.shape,dtype=np.int),
            "soup_cook_time_remaining": np.zeros(self.mdp.shape,dtype=np.int),  # val = remaining game ticks
            "soup_done": np.zeros(self.mdp.shape,dtype=np.int),                 # val = bool
            "dishes": np.zeros(self.mdp.shape,dtype=np.int),
            "onions": np.zeros(self.mdp.shape,dtype=np.int),
            # "tomatoes": np.zeros(self.mdp.shape),
        }
        for obj in state.all_objects_list:
            if obj.name == "soup":
                # get the ingredients into a {object: number} dictionary
                ingredients_dict = Counter(obj.ingredients)
                if obj.position in self.mdp.get_pot_locations():
                    if obj.is_idle:
                        # onions_in_pot and tomatoes_in_pot are used when the soup is idling, and ingredients could still be added
                        features["onions_in_pot"][obj.position] += ingredients_dict["onion"]
                        # features["tomatoes_in_pot"][obj.position] += ingredients_dict["tomato"]
                    else:
                        features["onions_in_soup"][obj.position] += ingredients_dict["onion"]
                        # features["tomatoes_in_soup"][obj.position] +=  ingredients_dict["tomato"]
                        features["soup_cook_time_remaining" ][obj.position] += obj.cook_time - obj._cooking_tick
                        if obj.is_ready:  features["soup_done"][obj.position] += 1
                else:
                    # If player soup is not in a pot, treat it like a soup that is cooked with remaining time 0
                    features["onions_in_soup"][obj.position] += ingredients_dict["onion"]
                    # features["tomatoes_in_soup"][obj.position] +=  ingredients_dict["tomato"]
                    features["soup_done"][obj.position] +=  1

            elif obj.name == "dish": features["dishes"][obj.position] += 1
            elif obj.name == "onion": features["onions"][obj.position] += 1
            # elif obj.name == "tomato": features["tomatoes"][obj.position] += 1
            else:  raise ValueError("Unrecognized object")
        return features

    def urgency_map_mask(self, state):
        features = {
            "urgency": np.zeros(self.mdp.shape,dtype=np.int)
        }
        if self.horizon - state.timestep < 40: features["urgency"] = np.ones(self.mdp.shape)
        return features

    # Vector Featurization #############################
    def featurize_vector(self,overcooked_state):
        """
        Takes in OvercookedState object and converts to observation index conducive to learning
        :param overcooked_state: OvercookedState object
        :return: observation feature vector for ego player
            ego_features: len = 2+4+3 = 9
                - pi_position: length 2 list of x,y position of player i
                - pi_orientation: length 4 one-hot-encoding of direction currently facing
                - pi_obj: length n_asset (3=|onion,plate,soup|) one-hot-encoding of object currently holding
            partner_features: len = 2+4+3 = 9
                - pj_position: length 2 list of x,y position of player i
                - pj_orientation: length 4 one-hot-encoding of direction currently facing
                - pj_obj: length n_asset (3=|onion,plate,soup|) one-hot-encoding of object currently holding
                TODO: Reduce feature space by removing orientation from partner player (pj)?
            world_features: n_counter+n_pot
                - counter_status: length n_counter of asset labels on counters {'nothing': 0, 'onion': 1, 'dish': 2, 'soup': 3}
                TODO: make 1-hot encoding of counter status/decouple corralation between counter and object?
                - pot_status: length n_pot of asset labels in pots {'0 onion': 0, '1 onion': 1, '2 onion': 2, '3 onion': 3, 'ready': 4}

            OTHER POSSIBLE FEATURES:
                - distance each static asset: d_asset = [dx,dy] (len = n_asset x 2) to contextualize goals
                    Assets: pot, ingrediant source, serving counter, trash, puddle/water
        """
        # ego_features = self.featurize_player(overcooked_state, self.my_index)
        # partner_features = self.featurize_player(overcooked_state, self.partner_index)
        # world_features = self.featurize_world(overcooked_state)
        # return np.concatenate([ego_features, partner_features, world_features]).astype(self.featurize_dtype)
        ego_features = self.featurize_player_vector(overcooked_state, self.my_index)
        world_features = self.featurize_world_vector(overcooked_state)
        return np.concatenate([ego_features, world_features]).astype(self.featurize_dtype)

    def featurize_player_vector(self, state, i):
        """
        Features:
            - pi_position: length 2 list of x,y position of player i
            - pi_orientation: length 4 one-hot-encoding of direction currently facing
            - pi_holding: length n_asset (3=|onion,plate,soup|) one-hot-encoding of object currently holding
        :param state: OvecookedState object
        :param i: which player to get features for
        :return: player_feature_vector
        """
        player_features = {}
        player = state.players[i]

        # Get position features ---------------
        player_features["p{}_position".format(i)] = np.array(player.position)

        # Get orientation features ---------------
        orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
        player_features["p{}_orientation".format(i)] = np.eye(4)[orientation_idx]

        # Get holding features (1-HOT)---------------
        obj = player.held_object
        if obj is None:
            held_obj_name = "none"
            player_features["p{}_objs".format(i)] = np.zeros(len(self.IDX_TO_OBJ))
        else:
            held_obj_name = obj.name
            obj_idx = self.OBJ_TO_IDX[held_obj_name]
            player_features["p{}_objs".format(i)] = np.eye(len(self.IDX_TO_OBJ))[obj_idx]

        # Create feature vector ---------------
        player_feature_vector = np.concatenate([player_features["p{}_position".format(i)],
                                                player_features["p{}_orientation".format(i)],
                                                player_features["p{}_objs".format(i)]])


        return player_feature_vector

    def featurize_world_vector(self, overcooked_state):
        """
        TODO: validate difference between full_but_not_cooking_pots and cooking_pots; doesnt it start automatically?
        Features:
            - counter_status: length n_counter of asset labels on counters {'nothing': 0, 'onion': 1, 'dish': 2, 'soup': 3}
            TODO: make 1-hot encoding of counter status/decouple corralation between counter and object?
            - pot_status: length n_pot of asset labels in pots
                {'empty': 0, '1 onion': 1, '2 onion': 2, '3 onion/cooking': 3, 'ready': 4}
                {'empty': 0, 'X items': X,..., 'cooking': 3, 'ready': 4}
            TODO: remove number of items in pot and instead just label as {'empty': 0, 'not_full':1, 'full':2, 'cooking':3, 'ready':4}

        # Other pot state info cmds
        # pot_states = self.mdp.get_pot_states(overcooked_state)
        # is_empty = int(pot_loc in self.mdp.get_empty_pots(pot_states))
        # is_full = int(pot_loc in self.mdp.get_full_but_not_cooking_pots(pot_states))
        # is_cooking = int(pot_loc in self.mdp.get_cooking_pots(pot_states))
        # is_ready = int(pot_loc in self.mdp.get_ready_pots(pot_states))
        # is_partially_ful = int(pot_loc in self.mdp.get_partially_full_pots(pot_states))

        :param overcooked_state: OvercookedState object
        :return: world_feature_vector
        """
        world_features = {}

        # get counter status feature vector (LABELED) ---------------
        # counter_locs = self.reachable_counters # self.mdp.get_counter_locations()
        # counter_labels = np.zeros(len(counter_locs))
        # counter_objs = self.mdp.get_counter_objects_dict(overcooked_state) # dictionary of pos:objects
        # for counter_loc, counter_obj in counter_objs.items():
        #     counter_labels[counter_locs.index(counter_loc)] = self.OBJ_TO_IDX[counter_obj]
        # world_features["counter_status"] = counter_labels

        # get counter status feature vector (1-Hot) ---------------
        counter_locs = self.reachable_counters  # self.mdp.get_counter_locations()
        counter_indicator_arr = np.zeros([len(counter_locs), len(self.IDX_TO_OBJ)])
        counter_objs = self.mdp.get_counter_objects_dict(overcooked_state)  # dictionary of pos:objects
        for counter_obj,counter_loc in counter_objs.items():
            iobj = self.OBJ_TO_IDX[counter_obj]
            icounter = counter_locs.index(counter_loc[0])
            counter_indicator_arr[icounter,iobj] = 1
        world_features["counter_status"] = counter_indicator_arr.flatten()

        # get pot status feature vector ---------------
        req_ingredients = self.mdp.recipe_config['num_items_for_soup'] # number of ingrediants before cooking
        pot_locs = self.mdp.get_pot_locations()
        pot_labels = np.zeros(len(pot_locs))
        for pot_index, pot_loc in enumerate(pot_locs):
            is_empty = not overcooked_state.has_object(pot_loc)
            if is_empty: pot_labels[pot_index] = 0
            else:
                soup = overcooked_state.get_object(pot_loc)
                if soup.is_ready:               pot_labels[pot_index]= req_ingredients + 1
                elif soup.is_cooking:           pot_labels[pot_index]= req_ingredients
                elif len(soup.ingredients) >0:  pot_labels[pot_index] = len(soup.ingredients)
                else: raise ValueError(f"Invalid pot state {soup}")


        world_features["pot_status"] = pot_labels

        # Create feature vector ---------------
        world_feature_vector = np.concatenate([world_features['counter_status'],
                                               world_features['pot_status']])
        return world_feature_vector

    def get_featurized_vector_shape(self):
        n_features = len(self.featurize(self.mdp.get_standard_start_state()))
        d_1hot = 2

        # Player features
        pos_dim = list(np.shape(self.mdp.terrain_mtx))[::-1]
        orientation_dim = [d_1hot for _ in range(len(Direction.ALL_DIRECTIONS))]
        hold_dim = [d_1hot for _ in range(len(self.IDX_TO_OBJ))]
        player_dim = pos_dim + orientation_dim + hold_dim

        # World features
        counter_dim = [d_1hot for _ in range(len(self.reachable_counters)* len(self.IDX_TO_OBJ))]
        pot_dim = [(self.mdp.recipe_config['num_items_for_soup']+2) * len(self.mdp.get_pot_locations())]
        world_dim = counter_dim + pot_dim

        # Full feature dim
        # feature_dim = player_dim + player_dim + world_dim
        feature_dim = player_dim + world_dim
        assert len(feature_dim) == n_features, f"Feature dim {len(feature_dim)} != n_features {n_features}"
        # return feature_dim

        return np.shape(feature_dim)

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


def verify_lossless_featurization(mdp, agent,state):
    """ Verify that the lossless featurization is correct"""
    primary_agent_idx = 0
    other_agent_idx = 1
    # Get baseline layer dict
    # ordered_player_features = ["player_{}_loc".format(primary_agent_idx),
    #                            "player_{}_loc".format(other_agent_idx)] + \
    #                           ["player_{}_orientation_{}".format(i, Direction.DIRECTION_TO_INDEX[d])
    #                            for i, d in product([primary_agent_idx, other_agent_idx],
    #                                                          Direction.ALL_DIRECTIONS)]
    #
    # base_map_features = ["pot_loc", "counter_loc", "onion_disp_loc", "dish_disp_loc", "serve_loc"]
    # variable_map_features = ["onions_in_pot", "onions_cook_time", "onion_soup_loc", "dishes", "onions"]
    # urgency_features = ["urgency"]
    # ordered_player_features = [
    #                               "player_{}_loc".format(primary_agent_idx),
    #                               "player_{}_loc".format(other_agent_idx),
    #                           ] + [
    #                               "player_{}_orientation_{}".format(i, Direction.DIRECTION_TO_INDEX[d])
    #                               for i, d in product(
    #         [primary_agent_idx, other_agent_idx],
    #         Direction.ALL_DIRECTIONS,
    #     )
    #                           ]
    # base_map_features = [
    #     "pot_loc",
    #     "counter_loc",
    #     "onion_disp_loc",
    #     "tomato_disp_loc",
    #     "dish_disp_loc",
    #     "serve_loc",
    #     "water_loc"
    # ]
    # variable_map_features = [
    #     "onions_in_pot",
    #     "tomatoes_in_pot",
    #     "onions_in_soup",
    #     "tomatoes_in_soup",
    #     "soup_cook_time_remaining",
    #     "soup_done",
    #     "dishes",
    #     "onions",
    #     "tomatoes",
    # ]
    # urgency_features = ["urgency"]
    # baseline_LAYERS = ordered_player_features  + base_map_features + variable_map_features + urgency_features
    baseline_LAYERS = ['player_0_loc', 'player_1_loc', 'player_0_orientation_0', 'player_0_orientation_1', 'player_0_orientation_2',
     'player_0_orientation_3', 'player_1_orientation_0', 'player_1_orientation_1', 'player_1_orientation_2',
     'player_1_orientation_3', 'pot_loc', 'counter_loc', 'onion_disp_loc', 'tomato_disp_loc', 'dish_disp_loc',
     'serve_loc', 'onions_in_pot', 'tomatoes_in_pot', 'onions_in_soup', 'tomatoes_in_soup', 'soup_cook_time_remaining',
     'soup_done', 'dishes', 'onions', 'tomatoes', 'urgency']

    # for name in baseline_LAYERS:
    #     print(f'LAYERS: {name}')
    # print(f'LAYERS Len: {len(baseline_LAYERS)}')

    # Get the lossless state encoding
    obs_baseline = mdp.lossless_state_encoding(state)[0]
    # Get the lossless state encoding from the agent
    obs_agent,obs_agent_keys = agent.featurize_mask(state,get_keys=True)

    # Check if all layers are present and correct
    # for i,layer in enumerate(baseline_LAYERS):
    #     if layer not in obs_agent_keys:
    #         print(f'lAYE not in Custom: {layer}')
    #         # raise AssertionError(f'Key not in baseline: {key}')
    #     else:
    #         i_agent = np.where(np.array(obs_agent_keys)==layer)[0][0]
    #         is_same = np.all(obs_baseline[...,i]==obs_agent[...,i_agent])
    #         print(f'Layer Check: [{layer} {obs_agent_keys[i_agent]}: {is_same}')
    #         if not is_same:
    #             print(f'Baseline:\n {obs_baseline[...,i]}')
    #             print(f'Obs:\n {obs_agent[...,i_agent]}')


    # Check if all are bool
    layer_names = [baseline_LAYERS,obs_agent_keys]
    for j,obs in enumerate([obs_baseline,obs_agent]):
        if not np.all(np.array(obs) <= 1):
            for i in range(obs.shape[-1]):
                a = obs[..., i]
                if not np.all(a <= 1) and not layer_names[j][i] in ['soup_cook_time_remaining','onions_in_pot','onions_in_soup']:
                    print(f'(MDP) Non-bool lossless encoding {np.max(a)} @ feature {i}:{baseline_LAYERS[i]}')
                    print(np.transpose(obs[..., i]))
                    raise AssertionError(f'({["MDP","Agent"][j]}) Non-bool lossless encoding {np.max(a)} @ feature {i}:{layer_names[j][i]}')
    # assert False
    return True
def test_update(n_tests = 100):
    model_config = {
        "obs_shape": [18, 5, 3],
        "n_actions": 6,
        "num_filters": 25,
        "num_convs": 3,  # CNN params
        "num_hidden_layers": 3,
        "size_hidden_layers": 32,
        "learning_rate": 1e-3,
        "n_mini_batch": 6,
        "minibatch_size": 32,
        "seed": 41
    }

    from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld
    from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv

    LAYOUT = "sanity_check2"; HORIZON = 500; ITERATIONS = 10_000

    # Generate MDP and environment----------------
    base_mdp = OvercookedGridworld.from_layout_name(LAYOUT)
    env = OvercookedEnv.from_mdp(base_mdp, horizon=HORIZON)
    # reply_memory = ReplayMemory(capacity=1000)
    # print(f'Layout: {LAYOUT} | Shape: {base_mdp.shape} | Iterations: {ITERATIONS}'')

    # Generate agents
    # model = DQN(**model_config)
    q_agent = SoloDeepQAgent(base_mdp,agent_index=0,model=DQN,config=model_config)
    stay_agent = StayAgent()
    # agents = [CustomQAgent(mdp, is_learning_agent=True, save_agent_file=True), StayAgent()]

    # base_env.reset()
    # state = base_env.state
    # action = Action.ALL_ACTIONS[0]
    # reward = 1
    # next_state = base_env.state
    # explore_decay_prog = 0.5
    # q_agent.update(state, action, reward, next_state, explore_decay_prog)


    for test in range(n_tests):
        env.reset()
        cum_reward = 0
        for t in count():
            state = env.state
            # q_agent.model.predict(q_agent.featurize(state)[np.newaxis])
            action1, _ = q_agent.action(state)
            action2, _ = stay_agent.action(state)
            joint_action = (action1, action2)
            next_state, reward, done, info = env.step(joint_action)  # what is joint-action info?
            reward += info["shaped_r_by_agent"][0]
            # if  info["shaped_r_by_agent"][0] != 0:
            #     print(f"Shaped reward {info['shaped_r_by_agent'][0]}")
            # q_agent.update(state, action1, reward, next_state, explore_decay_prog=total_updates / (HORIZON * ITERATIONS))
            # cum_reward += reward


            # reply_memory.push(q_agent.featurize(state), Action.ACTION_TO_INDEX[action1], reward, q_agent.featurize(next_state), done)
            replay_memory.append((q_agent.featurize(state), Action.ACTION_TO_INDEX[action1], reward, q_agent.featurize(next_state), done))
            # print(base_mdp.get_recipe_value(state))
            for obj in state.all_objects_list:
                if 'soup' ==obj.name:
                    if obj.is_cooking:
                        assert len(obj.ingredients)==3, 'Cooking soup should have 3 ingredients'
                        print('Cooking soup')
                    if obj.is_ready:
                        print('Soup is ready')
                    # print(f'Soup: {obj.ingredients} | is cooking={obj.is_cooking} | is ready={obj.is_ready} | is idle={obj.is_idle}')

            verify_lossless_featurization(base_mdp, q_agent, state)
            if done:
                break
        # print(f'Obs shape:{np.shape(q_agent.featurize(state))}')

        # experineces = reply_memory.sample(model_config['minibatch_size'])
        experineces = sample_experiences(model_config['minibatch_size'])
        q_agent.update(experineces)

        print(f"Test {test} passed")

def test_featurize():
    config = {
        "featurize_fn": "mask",
        "horizon": 500
    }
    from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld
    from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv

    LAYOUT = "sanity_check_3_onion"; HORIZON = 500; ITERATIONS = 10_000

    # Generate MDP and environment----------------
    # mdp_gen_params = {"layout_name": 'cramped_room_one_onion'}
    # mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_params)
    # env = OvercookedEnv(mdp_fn, horizon=HORIZON)
    base_mdp = OvercookedGridworld.from_layout_name(LAYOUT)
    base_env = OvercookedEnv.from_mdp(base_mdp, horizon=HORIZON)

    # Generate agents
    q_agent = SoloDeepQAgent(base_mdp,agent_index=0, save_agent_file=True,config=config)
    stay_agent = StayAgent()
    # agents = [CustomQAgent(mdp, is_learning_agent=True, save_agent_file=True), StayAgent()]

    base_env.reset()
    obs = q_agent.featurize(base_env.state)
    print(f'Featurized Shape:{q_agent.get_featurized_shape()}')
    print(f'Featurized Obs: {np.shape(obs)}:') # {obs}

    # print(f'Q-table Shape: {np.shape(q_agent.Q_table)}')
    # q_agent.Q_table[tuple(obs)][0] = 1
    # q = q_agent.Q_table[tuple(obs)]

    # print(f'Q-val ({np.shape(q)}): {q}')
    # Get the index of obs in the Q-table


if __name__ == "__main__":
    # test_featurize()
    test_update()

    #
    #
    #
