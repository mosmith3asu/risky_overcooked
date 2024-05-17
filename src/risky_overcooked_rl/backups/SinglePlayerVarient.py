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
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld
from itertools import product,count
import matplotlib.pyplot as plt
from develocorder import (
    LinePlot,
    Heatmap,
    FilteredLinePlot,
    DownsampledLinePlot,
    set_recorder,
    record,
    set_update_period,
    set_num_columns,
)



class CustomQAgent(Agent):
    """An agent randomly picks motion actions.
    Note: Does not perform interat actions, unless specified"""

    def __init__(self, mdp, agent_index, path_file= None, save_agent_file=False, load_learned_agent= False):
        # Learning Parameters ---------------

        self.my_index = agent_index
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

    def action(self, state,rationality=1.0):
        #TODO: player position is not enough to determine quality of action. What about if holding item, orientation/facing, ect...

        # return action to maximum Q table in setup
        # current_state = state.player_positions
        current_state = state.players_pos_and_or
        current_state_idx = self.valid_positions.index(current_state)
        #current_state_idx = np.where(self.valid_positions == current_state)[0]

        # Epsilon greedy ----------------
        if np.random.uniform(0,1) < self.exploration_proba:
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
                action_probs = np.exp(rationality*self.Q_table[current_state_idx, :]) / np.sum(np.exp(rationality*self.Q_table[current_state_idx, :]))
                action = Action.sample(action_probs)
                action_info = {"action_probs": action_probs}

            # Argmax action -----------
            # legal_actions = Action.ALL_ACTIONS
            # legal_actions_indices = np.array([Action.ACTION_TO_INDEX[motion_a] for motion_a in legal_actions])
            # legal_choice_idx = np.argmax(self.Q_table[current_state_idx, legal_actions_indices])
            #
            # # action_idx = np.argmax(self.Q_table[current_state_idx, :])
            # action_idx=legal_actions_indices[legal_choice_idx]
            #
            # action = self.valid_actions[action_idx]
            # action_probs = 1 #check what the action probs is supposed to be
            # action_info = {"action_probs": action_probs}

            # Check valid action -----------
            # action_set = self.mdp.get_actions(state)[self.my_index]
            # if action not in action_set:
            #     raise ValueError(f"Invalid action {action} for state {state}")

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
        #current_state = state.player_positions
        phy_state = state.players_pos_and_or
        next_phy_state = next_state.players_pos_and_or
        # phy_state= state.player_positions
        # next_phy_state = next_state.player_positions
        current_state_idx = self.valid_positions.index(phy_state)
        next_state_idx = self.valid_positions.index(next_phy_state)
        action_idx = self.valid_actions.index(action)
        #print(action_idx)
        #check is action or action_index
        self.Q_table[current_state_idx, action_idx] = (1 - self.lr) * self.Q_table[current_state_idx, action_idx] + self.lr * (
                    reward + self.gamma * max(self.Q_table[next_state_idx, :]))

        # self.exploration_proba = max(self.min_exploration_proba, np.exp(self.exploration_decreasing_decay*episode_length))

        self.exploration_proba = self.max_exploration_proba - explore_decay_prog*(self.max_exploration_proba - self.min_exploration_proba)

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
    # LAYOUT = "cramped_room_one_onion"
    LAYOUT = "sanity_check"; HORIZON = 500; ITERATIONS = 10_000

    # Logger ----------------
    # axis labels
    # set_recorder(labeled=LinePlot(xlabel="Step", ylabel="Score"))
    # additional filtered values (window filter)
    set_recorder(filtered=FilteredLinePlot(filter_size=10, xlabel="Iteration",ylabel=f"Score ({LAYOUT})")) #max_length=50,
    set_update_period(1)  # [seconds]

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


    total_updates = 0

    iter_rewards = []
    for iter in range(ITERATIONS):
        env.reset()

        cum_reward = 0
        for t in count():
            state = env.state
            action1, _ = q_agent.action(state)
            action2, _ = stay_agent.action(state)
            joint_action = (action1, action2)
            next_state, reward, done, _ = env.step(joint_action) # what is joint-action info?
            q_agent.update(state, action1, reward, next_state, explore_decay_prog=total_updates/(HORIZON*ITERATIONS))
            total_updates += 1
            # print(f"P(explore) {q_agent.exploration_proba}")
            cum_reward += reward
            if done:
                break

            # trajectory.append((s_t, a_t, r_t, done, info))

        # record(filtered=cum_reward)
        iter_rewards.append(cum_reward)

        # if len(iter_rewards) > 10 == 0:

        if len(iter_rewards) % 10 == 0:
            # Test policy #########################
            N_tests = 3
            test_reward = 0
            exploration_proba_OLD = q_agent.exploration_proba
            q_agent.exploration_proba = 0
            for test in range(N_tests):
                env.reset()
                for t in count():
                    state = env.state
                    featurized_state = env.featurize_state_mdp(state,num_pots=1)
                    print(f"State {np.shape(featurized_state)}")
                    action1, _ = q_agent.action(state,rationality=9)
                    action2, _ = stay_agent.action(state)
                    joint_action = (action1, action2)
                    next_state, reward, done, _ = env.step(joint_action)  # what is joint-action info?
                    total_updates += 1
                    # print(f"P(explore) {q_agent.exploration_proba}")
                    test_reward += reward

                    if done:
                        break

            record(filtered=test_reward / N_tests)
            q_agent.exploration_proba = exploration_proba_OLD
            print(f"Iteration {iter} complete | 10-Mean {np.mean(iter_rewards[-10:])} | Qminmax {q_agent.Q_table.min()} {q_agent.Q_table.max()} | P(explore) {q_agent.exploration_proba} | test reward= {test_reward / N_tests}")



    fig,ax = plt.subplots()
    ax.plot(iter_rewards)
    plt.show()

    # Generate agents
    # ptfp = "/home/rise/PycharmProjects/overcooked_ai/test/single_agent/2024-03-14-14-36-55"  # post training file path
    # q_agent = CustomQAgent(agent_eval.env.mlam, path_file=ptfp, load_learned_agent=False)


    # agent_pair = AgentPair(CustomRandomAgent(), CustomRandomAgent())
    # This is for training agent
    # single_q_agent_pair= AgentPair(CustomQAgent(agent_eval.env.mlam, is_learning_agent=True, save_agent_file=True), StayAgent())
    # trajectories_single_greedy_agent = agent_eval.evaluate_agent_pair(single_q_agent_pair, num_games=1)
    # print("Random pair rewards", trajectories_single_greedy_agent["ep_returns"])

    #This is for testing agent
    #TODO:check if you can access the file directly from above


    # q_agent = CustomQAgent(agent_eval.env.mlam, path_file=ptfp, load_learned_agent=True)
    # single_q_agent_pair= AgentPair(q_agent, StayAgent())
    #
    #
    # # single_q_agent_pair= AgentPair(CustomQAgent(agent_eval.env.mlam, path_file=ptfp, load_learned_agent=True), StayAgent())
    # trajectories_single_greedy_agent = agent_eval.evaluate_agent_pair(single_q_agent_pair, num_games=1)
    # print("Random pair rewards", trajectories_single_greedy_agent["ep_returns"])
    # print([q_agent.Q_table.min(),q_agent.Q_table.max()])
    # #TODO:check if you are able to get reward as well as visualization

    #SAVE the environment available in Agents.py
    #Test agent in the evaluate_agent_pair