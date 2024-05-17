import numpy as np
from risky_overcooked_py.mdp.actions import Action, Direction
from risky_overcooked_py.agents.agent import Agent, AgentPair,StayAgent, RandomAgent, GreedyHumanModel
import pickle
import datetime
import os

class SoloQAgent(Agent):
    """An agent randomly picks motion actions.
    Note: Does not perform interat actions, unless specified"""

    def __init__(self, mdp, agent_index,
                 path_file=None, save_agent_file=False, load_learned_agent=False,
                 config=None):


        # Learning Params ---------------
        self.max_exploration_proba = 0.9
        self.min_exploration_proba = 0.01
        self.exploration_proba = self.max_exploration_proba
        self.gamma = 0.95
        self.lr = 0.05
        self.rationality = 4.0
        self.load_learned_agent = load_learned_agent
        self.save_agent_file = save_agent_file
        self.tabular_dtype = np.float32

        # Featurize Params ---------------
        self.my_index = agent_index
        self.partner_index = int(not self.my_index)
        self.IDX_TO_OBJ = ["onion", "dish", "soup"]
        self.OBJ_TO_IDX = {o_name: idx for idx, o_name in enumerate(self.IDX_TO_OBJ)}
        self.reachable_counters = mdp.get_reachable_counters()
        self.featurize_dtype = np.int32

        # Create learning environment ---------------
        self.mdp = mdp
        self.Q_table = self.instantiate_tabular_Q()
        self.valid_actions = Action.ALL_ACTIONS

        # Save management ---------------
        # if self.save_agent_file:
        #     # use syspath, name and time
        #     output_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        #     os.makedirs("test/single_agent/%s" % output_name)
        #     # self.sim_out = open("./sim_outputs/%s/output.pkl" % output_name, "wb")
        #
        #     #
        #     # if self.capture:
        #     #     self.output_dir = "./sim_outputs/%s/video/" % output_name
        #     #     os.makedirs(self.output_dir)
        #
        #     self.learned_agent_path = "test/single_agent/%s" % output_name
        # elif self.load_learned_agent:
        #     self.learned_agent_path = path_file

        if config is not None:
            self.load_config(config)

    def load_config(self, config):
        for key, value in config.items():
            setattr(self, key, value)
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

    def action(self, state,enable_explore=True):
        # TODO: player position is not enough to determine quality of action. What about if holding item, orientation/facing, ect...

        # return action to maximum Q table in setup
        # current_state = state.player_positions
        # current_state = state.players_pos_and_or
        # current_state_idx = self.valid_positions.index(current_state)
        # current_state_idx = np.where(self.valid_positions == current_state)[0]

        # Epsilon greedy ----------------
        if np.random.uniform(0, 1) < self.exploration_proba and enable_explore:
            action_probs = np.zeros(Action.NUM_ACTIONS)
            legal_actions = Action.ALL_ACTIONS
            legal_actions_indices = np.array([Action.ACTION_TO_INDEX[motion_a] for motion_a in legal_actions])
            action_probs[legal_actions_indices] = 1 / len(legal_actions)

            action = Action.sample(action_probs)
            action_info = {"action_probs": action_probs}

        else:
            obs = self.featurize(state)
            qs = self.Q_table[tuple(obs)]
            # Boltzman Agent -----------
            if self.rationality == 'max':
                action_idx = np.argmax(qs)
                action = self.valid_actions[action_idx]
                action_probs = np.eye(Action.NUM_ACTIONS)[action_idx]
                action_info = {"action_probs": action_probs}
            else:
                # action_probs = np.exp(rationality * qs) / np.sum(np.exp(rationality * qs))
                action_probs = self.softmax(self.rationality * qs)
                action = Action.sample(action_probs)
                action_info = {"action_probs": action_probs}

        return action, action_info

    def actions(self, states, agent_indices):
        return (self.action(state) for state in states)

    def softmax(self, x):
        e_x = np.exp(x)
        return e_x / e_x.sum()
    def update(self, state, action, reward, next_state, explore_decay_prog,next_action=None):
        """

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param explore_decay_prog: [0,1] advc
        :return:
        """
        obs = self.featurize(state)
        next_obs = self.featurize(next_state)
        action_idx = self.valid_actions.index(action)
        try:
            if next_action is not None:
                if len(next_action) == Action.NUM_ACTIONS:
                    E_q_next = next_action@self.Q_table[tuple(next_obs)]
                else:
                    E_q_next = self.Q_table[tuple(next_obs)][self.valid_actions.index(next_action)]
                self.Q_table[tuple(obs)][action_idx] = (1 - self.lr) * self.Q_table[tuple(obs)][action_idx] \
                                                       + self.lr * (reward + self.gamma *  E_q_next)
            else:
                self.Q_table[tuple(obs)][action_idx] = (1 - self.lr) * self.Q_table[tuple(obs)][action_idx] \
                                                   + self.lr * (reward + self.gamma * max(self.Q_table[tuple(next_obs)]))
        except:
            print(f"Error updating Q-table: {len(np.shape(self.Q_table))} | {np.shape(obs)} | {np.shape(next_obs)} | {action_idx}")
            print(f"Q-table: {self.Q_table}")
            print(f"Obs: {obs}")
            print(f"Next Obs: {next_obs}")
            print(f"Action: {action_idx}")


        self.exploration_proba = self.max_exploration_proba - explore_decay_prog * (
                    self.max_exploration_proba - self.min_exploration_proba)

    def evaluate_state(self, state):
        return NotImplementedError()

    def instantiate_tabular_Q(self):
        obs_shape = self.get_featurized_shape()
        action_shape = [Action.NUM_ACTIONS]
        return np.zeros(obs_shape + action_shape, dtype=self.tabular_dtype)


    ########################################################
    # Featurization methods ################################
    ########################################################

    def featurize(self,overcooked_state):
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
        ego_features = self.featurize_player(overcooked_state, self.my_index)
        world_features = self.featurize_world(overcooked_state)
        return np.concatenate([ego_features, world_features]).astype(self.featurize_dtype)

    def featurize_player(self, state, i):
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

    def featurize_world(self, overcooked_state):
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

    def get_featurized_shape(self):
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
        return feature_dim

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


class SoloQAgent_ObsBoth(Agent):
    """An agent randomly picks motion actions.
    Note: Does not perform interat actions, unless specified"""

    def __init__(self, mdp, agent_index, path_file=None, save_agent_file=False, load_learned_agent=False):


        # Learning Params ---------------
        self.max_exploration_proba = 0.1
        self.min_exploration_proba = 0.01
        self.exploration_proba = self.max_exploration_proba
        self.gamma = 0.95
        self.lr = 0.05
        self.load_learned_agent = load_learned_agent
        self.save_agent_file = save_agent_file
        self.tabular_dtype = np.float32

        # Featurize Params ---------------
        self.my_index = agent_index
        self.partner_index = int(not self.my_index)
        self.IDX_TO_OBJ = ["onion", "dish", "soup"]
        self.OBJ_TO_IDX = {o_name: idx for idx, o_name in enumerate(self.IDX_TO_OBJ)}
        self.reachable_counters = mdp.get_reachable_counters()
        self.featurize_dtype = np.int32

        # Create learning environment ---------------
        self.mdp = mdp
        self.Q_table = self.instantiate_tabular_Q()
        self.valid_actions = Action.ALL_ACTIONS

        # Save management ---------------
        # if self.save_agent_file:
        #     # use syspath, name and time
        #     output_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        #     os.makedirs("test/single_agent/%s" % output_name)
        #     # self.sim_out = open("./sim_outputs/%s/output.pkl" % output_name, "wb")
        #
        #     #
        #     # if self.capture:
        #     #     self.output_dir = "./sim_outputs/%s/video/" % output_name
        #     #     os.makedirs(self.output_dir)
        #
        #     self.learned_agent_path = "test/single_agent/%s" % output_name
        # elif self.load_learned_agent:
        #     self.learned_agent_path = path_file


    ########################################################
    # Learning/Performance methods #########################
    ########################################################

    def action(self, state, rationality=1.0,enable_explore=True):
        # TODO: player position is not enough to determine quality of action. What about if holding item, orientation/facing, ect...

        # return action to maximum Q table in setup
        # current_state = state.player_positions
        # current_state = state.players_pos_and_or
        # current_state_idx = self.valid_positions.index(current_state)
        # current_state_idx = np.where(self.valid_positions == current_state)[0]

        # Epsilon greedy ----------------
        if np.random.uniform(0, 1) < self.exploration_proba and enable_explore:
            action_probs = np.zeros(Action.NUM_ACTIONS)
            legal_actions = Action.ALL_ACTIONS
            legal_actions_indices = np.array([Action.ACTION_TO_INDEX[motion_a] for motion_a in legal_actions])
            action_probs[legal_actions_indices] = 1 / len(legal_actions)

            action = Action.sample(action_probs)
            action_info = {"action_probs": action_probs}

        else:
            obs = self.featurize(state)
            qs = self.Q_table[tuple(obs)]
            # Boltzman Agent -----------
            if rationality == 'max':
                action_idx = np.argmax(qs)
                action = self.valid_actions[action_idx]
                action_probs = np.eye(Action.NUM_ACTIONS)[action_idx]
                action_info = {"action_probs": action_probs}
            else:
                # action_probs = np.exp(rationality * qs) / np.sum(np.exp(rationality * qs))
                action_probs = self.softmax(rationality * qs)
                assert np.isclose(np.sum(action_probs), 1), f"Action probs not normalized: {action_probs}"
                try:
                    action = Action.sample(action_probs)
                except:
                    print(f"Error sampling action: {action_probs}")
                    print(f"Qs: {qs}")
                    print(f"Obs: {obs}")
                    print(f"Action Probs: {action_probs}")
                    raise
                action_info = {"action_probs": action_probs}

        return action, action_info

    def actions(self, states, agent_indices):
        return (self.action(state) for state in states)

    def softmax(self, x):
        e_x = np.exp(x)
        return e_x / e_x.sum()
    def update(self, state, action, reward, next_state, explore_decay_prog):
        """

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param explore_decay_prog: [0,1] advc
        :return:
        """
        obs = self.featurize(state)
        next_obs = self.featurize(next_state)
        action_idx = self.valid_actions.index(action)
        try:
            self.Q_table[tuple(obs)][action_idx] = (1 - self.lr) * self.Q_table[tuple(obs)][action_idx] \
                                                   + self.lr * (reward + self.gamma * max(self.Q_table[tuple(next_obs)]))
        except:
            print(f"Error updating Q-table: {np.shape(self.Q_table)} | {np.shape(obs)} | {np.shape(next_obs)} | {action_idx}")
            print(f"Q-table: {self.Q_table}")
            print(f"Obs: {obs}")
            print(f"Next Obs: {next_obs}")
            print(f"Action: {action_idx}")

        self.exploration_proba = self.max_exploration_proba - explore_decay_prog * (
                    self.max_exploration_proba - self.min_exploration_proba)

    def evaluate_state(self, state):
        return NotImplementedError()

    def instantiate_tabular_Q(self):
        obs_shape = self.get_featurized_shape()
        action_shape = [Action.NUM_ACTIONS]
        return np.zeros(obs_shape + action_shape, dtype=self.tabular_dtype)


    ########################################################
    # Featurization methods ################################
    ########################################################

    def featurize(self,overcooked_state):
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
        ego_features = self.featurize_player(overcooked_state, self.my_index)
        partner_features = self.featurize_player(overcooked_state, self.partner_index)
        world_features = self.featurize_world(overcooked_state)
        return np.concatenate([ego_features, partner_features, world_features]).astype(self.featurize_dtype)

    def featurize_player(self, state, i):
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

    def featurize_world(self, overcooked_state):
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
        for counter_loc, counter_obj in counter_objs.items():
            iobj = self.OBJ_TO_IDX[counter_obj]
            icounter = counter_locs.index(counter_loc)
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

    def get_featurized_shape(self):
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
        feature_dim = player_dim + player_dim + world_dim
        assert len(feature_dim) == n_features, f"Feature dim {len(feature_dim)} != n_features {n_features}"
        return feature_dim

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


def test_update():

    from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld
    from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv

    LAYOUT = "sanity_check_3_onion"; HORIZON = 500; ITERATIONS = 10_000

    # Generate MDP and environment----------------
    base_mdp = OvercookedGridworld.from_layout_name(LAYOUT)
    base_env = OvercookedEnv.from_mdp(base_mdp, horizon=HORIZON)

    # Generate agents
    q_agent = SoloQAgent(base_mdp,agent_index=0, save_agent_file=True)
    stay_agent = StayAgent()
    # agents = [CustomQAgent(mdp, is_learning_agent=True, save_agent_file=True), StayAgent()]

    base_env.reset()
    state = base_env.state
    action = Action.ALL_ACTIONS[0]
    reward = 1
    next_state = base_env.state
    explore_decay_prog = 0.5
    q_agent.update(state, action, reward, next_state, explore_decay_prog)


def test_featurize():
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
    q_agent = SoloQAgent(base_mdp,agent_index=0, save_agent_file=True)
    stay_agent = StayAgent()
    # agents = [CustomQAgent(mdp, is_learning_agent=True, save_agent_file=True), StayAgent()]

    base_env.reset()
    obs = q_agent.featurize(base_env.state)
    print(f'Featurized Shape:{q_agent.get_featurized_shape()}')
    print(f'Featurized Obs: {np.shape(obs)}: {obs}')

    print(f'Q-table Shape: {np.shape(q_agent.Q_table)}')
    q_agent.Q_table[tuple(obs)][0] = 1
    q = q_agent.Q_table[tuple(obs)]

    print(f'Q-val ({np.shape(q)}): {q}')
    # Get the index of obs in the Q-table


if __name__ == "__main__":
    test_featurize()
    # test_update()

    #
    #
    #
