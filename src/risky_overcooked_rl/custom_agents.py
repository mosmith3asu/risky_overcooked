import numpy as np
from risky_overcooked_py.mdp.actions import Action, Direction
from risky_overcooked_py.agents.agent import Agent, AgentPair,StayAgent, RandomAgent, GreedyHumanModel
import pickle
import datetime
import os

class SoloQAgent(Agent):
    """An agent randomly picks motion actions.
    Note: Does not perform interat actions, unless specified"""

    def __init__(self, mdp, agent_index, path_file=None, save_agent_file=False, load_learned_agent=False):
        # Learning Parameters ---------------

        self.my_index = agent_index
        self.partner_index = int(not self.my_index)

        # self.exploration_decreasing_decay = 0.001
        self.max_exploration_proba = 0.25
        self.min_exploration_proba = 0.01
        self.exploration_proba = self.max_exploration_proba
        self.gamma = 0.95
        self.lr = 0.01
        self.load_learned_agent = load_learned_agent
        self.save_agent_file = save_agent_file


        # Create state-action space
        self.mdp = mdp



        # initialize helpers
        # self.instantiate_asset_placements()
        self.valid_player_positions_and_orientations = self.mdp.get_valid_joint_player_positions_and_orientations()

        self.valid_positions = self.mdp.get_valid_joint_player_positions_and_orientations()
        # Old ------
        # self.mdp.get_valid_player_positions_and_orientations()
        # self.valid_orientations = self.mdp.get_valid_player_orientations()
        # self.valid_position_1 = self.mdp.get_valid_player_positions()
        # self.valid_positions = []
        # for item in product(self.valid_position_1, repeat=2):
        #     self.valid_positions.append(item)
        # Joint Action
        # self.valid_actions = []
        # for item in product(Action.ALL_ACTIONS, repeat=2):
        #     self.valid_actions.append(item)
        # Single Action
        self.valid_actions = Action.ALL_ACTIONS

        # Create q-table
        # self.Q_table = np.zeros((len(self.valid_positions), len(self.valid_actions)))
        self.Q_table = np.zeros((len(self.valid_positions), len(self.valid_actions)))

        if self.save_agent_file:
            # use syspath, name and time
            output_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            os.makedirs("test/single_agent/%s" % output_name)
            # self.sim_out = open("./sim_outputs/%s/output.pkl" % output_name, "wb")

            #
            # if self.capture:
            #     self.output_dir = "./sim_outputs/%s/video/" % output_name
            #     os.makedirs(self.output_dir)

            self.learned_agent_path = "test/single_agent/%s" % output_name
        elif self.load_learned_agent:
            self.learned_agent_path = path_file

    def action(self, state, rationality=1.0):
        # TODO: player position is not enough to determine quality of action. What about if holding item, orientation/facing, ect...

        # return action to maximum Q table in setup
        # current_state = state.player_positions
        current_state = state.players_pos_and_or
        current_state_idx = self.valid_positions.index(current_state)
        # current_state_idx = np.where(self.valid_positions == current_state)[0]

        # Epsilon greedy ----------------
        if np.random.uniform(0, 1) < self.exploration_proba:
            action_probs = np.zeros(Action.NUM_ACTIONS)
            legal_actions = Action.ALL_ACTIONS
            legal_actions_indices = np.array([Action.ACTION_TO_INDEX[motion_a] for motion_a in legal_actions])
            action_probs[legal_actions_indices] = 1 / len(legal_actions)

            action = Action.sample(action_probs)
            action_info = {"action_probs": action_probs}

            # Check valid action
            # action_set = self.mdp.get_actions(state)[self.my_index]
            # if action not in action_set:
            #     raise ValueError(f"Invalid action {action} for state {state}")

        else:
            # Boltzman Agent -----------
            if rationality == 'max':
                action_idx = np.argmax(self.Q_table[current_state_idx, :])
                action = self.valid_actions[action_idx]
                action_probs = 1
                action_info = {"action_probs": action_probs}
            else:
                action_probs = np.exp(rationality * self.Q_table[current_state_idx, :]) / np.sum(
                    np.exp(rationality * self.Q_table[current_state_idx, :]))
                action = Action.sample(action_probs)
                action_info = {"action_probs": action_probs}

        return action, action_info

    def actions(self, states, agent_indices):
        return (self.action(state) for state in states)

    def update(self, state, action, reward, next_state, explore_decay_prog):
        """

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param explore_decay_prog: [0,1] advc
        :return:
        """
        # print("update q-value")
        # current_state = state.player_positions
        phy_state = state.players_pos_and_or
        next_phy_state = next_state.players_pos_and_or
        # phy_state= state.player_positions
        # next_phy_state = next_state.player_positions
        current_state_idx = self.valid_positions.index(phy_state)
        next_state_idx = self.valid_positions.index(next_phy_state)
        action_idx = self.valid_actions.index(action)
        # print(action_idx)
        # check is action or action_index
        self.Q_table[current_state_idx, action_idx] = (1 - self.lr) * self.Q_table[
            current_state_idx, action_idx] + self.lr * (
                                                              reward + self.gamma * max(
                                                          self.Q_table[next_state_idx, :]))

        # self.exploration_proba = max(self.min_exploration_proba, np.exp(self.exploration_decreasing_decay*episode_length))

        self.exploration_proba = self.max_exploration_proba - explore_decay_prog * (
                    self.max_exploration_proba - self.min_exploration_proba)

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

    def featurize(self,overcooked_state):
        """

        Takes in OvercookedState object and converts to observation index conducive to learning
        :param overcooked_state: OvercookedState object
        :return: observation feature vector for ego player
            player_i_features: ego agent features
                - pi_position: length 2 list of x,y position of player i
                - pi_orientation: length 4 one-hot-encoding of direction currently facing
                - pi_obj: length n_asset (3) one-hot-encoding of object currently holding

            player_j_features: other partner player features
                - pj_position: length 2 list of x,y position of player i
                - pj_orientation: length 4 one-hot-encoding of direction currently facing
                - pj_obj: length n_asset (3) one-hot-encoding of object currently holding
                TODO: Reduce feature space by removing orientation from partner player (pj)?
            world_features:
                - counter_status: length n_counter of asset labels on counters {'nothing': 0, 'onion': 1, 'dish': 2, 'soup': 3}
                TODO: make 1-hot encoding of counter status/decouple corralation between counter and object?
                - pot_status: length n_pot of asset labels in pots {'0 onion': 0, '1 onion': 1, '2 onion': 2, '3 onion': 3, 'ready': 4}

        """
        player_i_features = self.featureize_player(overcooked_state, self.my_index)

    ######### HELPER FUNCTS MODIFYING ENVIRONMENT STATE #########
    def featureize_player(self, state, i):
        """
        Features:
            - pi_position: length 2 list of x,y position of player i
            - pi_orientation: length 4 one-hot-encoding of direction currently facing
            - pi_holding: length n_asset (3) one-hot-encoding of object currently holding
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

        # Get holding features ---------------
        IDX_TO_OBJ = ["onion",  "dish", "soup"]
        OBJ_TO_IDX = {o_name: idx for idx, o_name in enumerate(IDX_TO_OBJ)}
        obj = player.held_object
        if obj is None:
            held_obj_name = "none"
            player_features["p{}_objs".format(i)] = np.zeros(len(IDX_TO_OBJ))
        else:
            held_obj_name = obj.name
            obj_idx = OBJ_TO_IDX[held_obj_name]
            player_features["p{}_objs".format(i)] = np.eye(len(IDX_TO_OBJ))[obj_idx]

        # Create feature vector ---------------
        player_feature_vector = np.concatenate([player_features["p{}_position".format(i)],
                                                player_features["p{}_orientation".format(i)],
                                                player_features["p{}_objs".format(i)]])

        return player_feature_vector
    def featureize_world(self, overcooked_state):
        """
        Features:
            - counter_status: length n_counter of asset labels on counters {'nothing': 0, 'onion': 1, 'dish': 2, 'soup': 3}
            TODO: make 1-hot encoding of counter status/decouple corralation between counter and object?
            - pot_status: length n_pot of asset labels in pots {'0 onion': 0, '1 onion': 1, '2 onion': 2, '3 onion': 3, 'ready': 4}

        :param overcooked_state:
        :return:
        """
        world_features = {}

        # get counter status ---------------
        counter_objects = self.mdp.get_counter_objects_dict(overcooked_state)




        pot_states = self.mdp.get_pot_states(overcooked_state)

    def instantiate_asset_placements(self):
        empty_counters = self.mdp.get_empty_counter_locations()










if __name__ == "__main__":
    from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld
    from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv

    LAYOUT = "sanity_check"; HORIZON = 500; ITERATIONS = 10_000

    # Generate MDP and environment----------------
    # mdp_gen_params = {"layout_name": 'cramped_room_one_onion'}
    # mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_params)
    # env = OvercookedEnv(mdp_fn, horizon=HORIZON)
    mdp = OvercookedGridworld.from_layout_name(LAYOUT)
    env = OvercookedEnv.from_mdp(mdp, horizon=HORIZON)


    # Generate agents
    q_agent = CustomQAgent(mdp,agent_index=0, save_agent_file=True)
    stay_agent = StayAgent()
    # agents = [CustomQAgent(mdp, is_learning_agent=True, save_agent_file=True), StayAgent()]

    env.reset()
    q_agent.featurize(env.state)
