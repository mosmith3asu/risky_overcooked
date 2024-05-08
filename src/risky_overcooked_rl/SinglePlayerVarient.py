import numpy as np
from risky_overcooked_py.mdp.actions import Action, Direction
from risky_overcooked_py.agents.benchmarking import AgentEvaluator,LayoutGenerator
from risky_overcooked_py.agents.agent import Agent, AgentPair,StayAgent, RandomAgent, GreedyHumanModel
# from risky_overcooked_py.mdp.overcooked_mdp import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
import pickle
import datetime
import os
from tempfile import TemporaryFile






class CustomQAgent(Agent):
    """An agent randomly picks motion actions.
    Note: Does not perform interat actions, unless specified"""

    def __init__(self, mlam, path_file= None, is_learning_agent=False, save_agent_file=False, load_learned_agent= False):
        #check state space
        self.mlam= mlam # mid level planner
        self.mdp = self.mlam.mdp
        #check q table shape and the matrix shape
        self.valid_position_1 = self.mdp.get_valid_player_positions()
        from itertools import product
        self.valid_positions = []
        self.valid_actions = []
        for item in product(self.valid_position_1, repeat=2):
            self.valid_positions.append(item)
        for item in product(Action.ALL_ACTIONS, repeat=2):
            self.valid_actions.append(item)
        #self.Q_table = np.zeros((len(self.valid_positions), Action.NUM_ACTIONS))
        self.Q_table = np.zeros((len(self.valid_positions), len(self.valid_actions)))
        self.exploration_proba = 0.9
        self.exploration_decreasing_decay = 0.001
        self.min_exploration_proba = 0.01
        self.gamma = 0.9
        self.lr = 0.1
        self.is_learning_agent = is_learning_agent
        self.save_agent_file=save_agent_file
        self.load_learned_agent= load_learned_agent
        if self.save_agent_file:
            #use syspath, name and time
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

    def action(self, state):
        # return action to maximum Q table in setup
        current_state = state.player_positions
        current_state_idx = self.valid_positions.index(current_state)
        #current_state_idx = np.where(self.valid_positions == current_state)[0]
        if np.random.uniform(0,1) < self.exploration_proba:
            action_probs = np.zeros(Action.NUM_ACTIONS)
            legal_actions = Action.ALL_ACTIONS
            legal_actions_indices = np.array([Action.ACTION_TO_INDEX[motion_a] for motion_a in legal_actions])
            action_probs[legal_actions_indices] = 1 / len(legal_actions)
            return Action.sample(action_probs), {"action_probs": action_probs}

        else:
            action_idx = np.argmax(self.Q_table[current_state_idx,:])


            action_probs = 1 #check what the action probs is supposed to be
            # use action_probs for the boltzman rationality

            #print("q learning agent",  current_state, self.valid_actions[action_idx][0])
            return self.valid_actions[action_idx][0], {"action_probs": action_probs}
            #return Action.ALL_ACTIONS[action_idx], {"action_probs": action_probs}


    def actions(self, states, agent_indices):
        return (self.action(state) for state in states)

    def update(self, state, action, reward, next_state, episode_length):
        print("update q-value")
        #current_state = state.player_positions
        phy_state= state.player_positions
        next_phy_state = next_state.player_positions
        current_state_idx = self.valid_positions.index(phy_state)
        next_state_idx = self.valid_positions.index(next_phy_state)
        action_idx = self.valid_actions.index(action)
        #print(action_idx)
        #check is action or action_index
        self.Q_table[current_state_idx, action_idx] = (1 - self.lr) * self.Q_table[current_state_idx, action_idx] + self.lr * (
                    reward + self.gamma * max(self.Q_table[next_state_idx, :]))

        self.exploration_proba = max(self.min_exploration_proba, np.exp(self.exploration_decreasing_decay*episode_length))
        #print(episode_length)
    # def save_agent(self):
    #     qtable = TemporaryFile()
    #     np.save(f"%s/qtable.npy", self.learned_agent_path, self.Q_table)
    #     #check how to save agent
    #
    # def load_agent(self,npy_path):
    #     #self.Q_table = np.load(f"qtables/{i}-qtable.npy")
    #     self.Q_table = np.load(npy_path)

    def save_agent(self, filename):
        """
        Save the Q-table to a file.

        Args:
            filename (str): The name of the file to save the Q-table to.

        Returns:
            None
        """
        #filename = os.path.join(os.path.dirname(__file__), filename)
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



if __name__ == "__main__":
    horizon = 100000

    # Generate MDP and environment
    mdp_gen_params = {"layout_name": 'cramped_room_one_onion'}
    mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_params)
    env = OvercookedEnv(mdp_fn, horizon=horizon)

    # Generate agents
    ptfp = "/home/rise/PycharmProjects/overcooked_ai/test/single_agent/2024-03-14-14-36-55"  # post training file path
    q_agent = CustomQAgent(agent_eval.env.mlam, path_file=ptfp, load_learned_agent=False)


    # agent_pair = AgentPair(CustomRandomAgent(), CustomRandomAgent())
    # This is for training agent
    # single_q_agent_pair= AgentPair(CustomQAgent(agent_eval.env.mlam, is_learning_agent=True, save_agent_file=True), StayAgent())
    # trajectories_single_greedy_agent = agent_eval.evaluate_agent_pair(single_q_agent_pair, num_games=1)
    # print("Random pair rewards", trajectories_single_greedy_agent["ep_returns"])

    #This is for testing agent
    #TODO:check if you can access the file directly from above


    q_agent = CustomQAgent(agent_eval.env.mlam, path_file=ptfp, load_learned_agent=True)
    single_q_agent_pair= AgentPair(q_agent, StayAgent())


    # single_q_agent_pair= AgentPair(CustomQAgent(agent_eval.env.mlam, path_file=ptfp, load_learned_agent=True), StayAgent())
    trajectories_single_greedy_agent = agent_eval.evaluate_agent_pair(single_q_agent_pair, num_games=1)
    print("Random pair rewards", trajectories_single_greedy_agent["ep_returns"])
    print([q_agent.Q_table.min(),q_agent.Q_table.max()])
    #TODO:check if you are able to get reward as well as visualization

    #SAVE the environment available in Agents.py
    #Test agent in the evaluate_agent_pair