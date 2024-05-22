import numpy as np
from risky_overcooked_py.mdp.actions import Action, Direction
from risky_overcooked_py.agents.benchmarking import AgentEvaluator,LayoutGenerator
from risky_overcooked_py.agents.agent import Agent, AgentPair,StayAgent, RandomAgent, GreedyHumanModel
from risky_overcooked_rl.utils.custom_deep_agents import SoloDeepQAgent
from risky_overcooked_rl.utils.deep_models import DQN,ReplayMemory
from risky_overcooked_rl.utils.rl_logger import FunctionTimer
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
debug = False
config = {
        'ALGORITHM': 'solo_dqn_egreedy',
        "seed": 41,

        # Env Params
        'LAYOUT': "sanity_check2",
        'HORIZON': 250,
        'ITERATIONS': 1_000,
        'init_reward_shaping_scale': 1.0,
        "obs_shape": [18, 5, 3],
        "n_actions": 6,

        # Learning Params
        'max_exploration_proba': 0.9,
        'min_exploration_proba': 0.01,
        'gamma': 0.95,
        "learning_rate": 1e-3,
        # "learning_rate": 1e-2,
        'rationality': 'max', # Boltzmann sampling
        "num_filters": 25,  # CNN params
        "num_convs": 3,  # CNN params
        "num_hidden_layers": 3,  # MLP params
        "size_hidden_layers": 32,  # MLP params
        # "n_mini_batch": 1,
        "n_mini_batch": 32,
        "minibatch_size": 32,
        "replay_memory_size": 5000,

        # Evaluation Params
        'test_interval': 10,
        'N_tests': 1,
        'logger_filter_size': 10,
        'logger_update_period': 1,  # [seconds]
    }

# config = {
#         'ALGORITHM': 'solo_dqn',
#         "seed": 41,
#
#         # Env Params
#         'LAYOUT': "sanity_check2",
#         'HORIZON': 200,
#         'ITERATIONS': 3_000,
#         'init_reward_shaping_scale': 1.0,
#         "obs_shape": [18, 5, 3],
#         "n_actions": 6,
#
#         # Learning Params
#         'max_exploration_proba': 0.75,
#         'min_exploration_proba': 0.01,
#         'gamma': 0.95,
#         "learning_rate": 1e-3,
#         # "learning_rate": 1e-2,
#         'rationality': 10.0, # Boltzmann sampling
#         "num_filters": 25,  # CNN params
#         "num_convs": 3,  # CNN params
#         "num_hidden_layers": 3,  # MLP params
#         "size_hidden_layers": 32,  # MLP params
#         # "n_mini_batch": 1,
#         "n_mini_batch": 16,
#         "minibatch_size": 32,
#         "replay_memory_size": 5000,
#
#         # Evaluation Params
#         'test_interval': 10,
#         'N_tests': 5,
#         'logger_filter_size': 10,
#         'logger_update_period': 1,  # [seconds]
#     }



if __name__ == "__main__":
    ALGORITHM = config['ALGORITHM']
    LAYOUT = config['LAYOUT']
    HORIZON = config['HORIZON']
    ITERATIONS = config['ITERATIONS']
    N_tests = config['N_tests']
    test_interval = config['test_interval']
    init_reward_shaping_scale = config['init_reward_shaping_scale']
    min_explore = config['min_exploration_proba']
    max_explore = config['max_exploration_proba']
    n_mini_batch = config['n_mini_batch']
    minibatch_size = config['minibatch_size']

    # Logger ----------------
    set_recorder(reward=FilteredLinePlot(filter_size=config['logger_filter_size'],
                                           xlabel="Iteration",
                                           ylabel=f"Score ({LAYOUT}|{ALGORITHM})"))
    set_recorder(shaped_reward=FilteredLinePlot(filter_size=config['logger_filter_size'],
                                           xlabel="Iteration",
                                           ylabel=f"Shaped Score ({LAYOUT}|{ALGORITHM})"))
    set_update_period(config['logger_update_period'])  # [seconds]
    # ftimer = FunctionTimer()

    # Generate MDP and environment----------------
    mdp = OvercookedGridworld.from_layout_name(LAYOUT)
    env = OvercookedEnv.from_mdp(mdp, horizon=HORIZON)
    model = DQN(**config)
    replay_memory = ReplayMemory(config["replay_memory_size"])

    # Generate agents
    q_agent = SoloDeepQAgent(mdp,agent_index=0,model=model,config=config)
    stay_agent = StayAgent()



    iter_rewards = []
    for iter in range(ITERATIONS):
        # ftimer.clear()
        q_agent.exploration_proba = max_explore-(iter/ITERATIONS)*(max_explore-min_explore)
        r_shape_scale = init_reward_shaping_scale-(iter/ITERATIONS)*(init_reward_shaping_scale-0)


        env.reset()

        # Simulate Episode ----------------
        cum_reward = 0

        for t in count():
            state = env.state
            action1, _ = q_agent.action(state)
            action2, _ = stay_agent.action(state)
            joint_action = (action1, action2)
            next_state, reward, done, info = env.step(joint_action) # what is joint-action info?
            reward += r_shape_scale*info["shaped_r_by_agent"][0]
            cum_reward += reward
            replay_memory.push(q_agent.featurize(state), Action.ACTION_TO_INDEX[action1], reward,
                                  q_agent.featurize(next_state), done)
            if done:  break
        print(f"Iteration {iter} | cumulative train reward: {cum_reward} |  P(explore) {q_agent.exploration_proba} ")

        # -------------------------------
        for _ in range(n_mini_batch):
            # experiences = ftimer('sample',  replay_memory.sample, minibatch_size)
            # ftimer('update',q_agent.update,experiences)
            experiences = replay_memory.sample(minibatch_size)
            q_agent.update(experiences)

        # Test policy ---------------------

        if iter % test_interval == 0:
            if debug: print('Test policy')
            test_reward = 0
            test_shaped_reward = 0
            for test in range(N_tests):
                env.reset()
                for t in count():
                    if debug: print(f'Test policy: test {test}, t {t}')
                    state = env.state
                    action1, _ = q_agent.action(state,enable_explore=False)
                    # action1, _ = ftimer('action',q_agent.action,state, enable_explore=False)
                    # obs = ftimer('featurize', q_agent.featurize, state)[np.newaxis]
                    # # action1, _ = ftimer('action',q_agent.action,obs1, enable_explore=False)
                    # qs = ftimer('predict',q_agent.model.predict,obs).flatten()
                    # # qs = q_agent.model.predict_timed(obs).flatten()
                    # action_probs = ftimer('softmax',q_agent.softmax,q_agent.rationality * qs)
                    # action1 = Action.sample(action_probs)
                    # action_info = {"action_probs": action_probs}

                    action2, _ = stay_agent.action(state)
                    joint_action = (action1, action2)
                    next_state, reward, done, info = env.step(joint_action)
                    test_reward += reward
                    test_shaped_reward += reward + info["shaped_r_by_agent"][0]
                    if done: break
            record(reward=test_reward / N_tests)
            record(shaped_reward=test_shaped_reward / N_tests)
            print(f"\nTest: | nTests= {N_tests} | Ave Reward = {test_reward / N_tests} | Ave Shaped Reward = {test_shaped_reward / N_tests}\n")
            # ftimer.report()
        # -------------------------------

        # if iter % 100 == 0:
        #     print(f"Iteration {iter} complete | 10-Mean {np.mean(iter_rewards[-10:])} | Qminmax {q_agent.Q_table.min()} {q_agent.Q_table.max()} | P(explore) {q_agent.exploration_proba} | test reward= {test_reward / N_tests}")



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