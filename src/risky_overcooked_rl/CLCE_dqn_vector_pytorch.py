import numpy as np
from risky_overcooked_py.agents.agent import Agent, AgentPair,StayAgent, RandomAgent, GreedyHumanModel
from risky_overcooked_rl.utils.custom_deep_agents import SoloDeepQAgent
from risky_overcooked_rl.utils.deep_models_pytorch import ReplayMemory,DQN_vector_feature,optimize_model,soft_update,device
from risky_overcooked_rl.utils.state_utils import StartStateManager
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld,SoupState
from itertools import count
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

import torch
import torch.optim as optim
import math

debug = False
config = {
        'ALGORITHM': 'solo_dqn_vector_egreedy',
        "seed": 41,

        # Env Params ############################
        'LAYOUT': "cramped_room_CLCE", 'HORIZON': 200, 'ITERATIONS': 10_000,
        # 'LAYOUT': "risky_cramped_room_single", 'HORIZON': 200, 'ITERATIONS': 10_000,
        # 'LAYOUT': "cramped_room_single", 'HORIZON': 200, 'ITERATIONS': 10_000,
        # 'LAYOUT': "sanity_check3_single", 'HORIZON': 400, 'ITERATIONS': 500,
        # 'LAYOUT': "sanity_check4_single", 'HORIZON': 300, 'ITERATIONS': 5_000,
        # 'LAYOUT': "sanity_check3",
        # 'LAYOUT': "sanity_check2",
        'n_players': 2,                     # number of players in the environment
        'n_player_features': 9,             # number of features in the player vector encoding (FIXED)
        "n_actions": 36,                    # number of actions in the action space
        "obs_shape": None,                  # set dynamically during runtime (leave as None)
        "perc_random_start": 0.80,          # percentage of iterations to randomize start state
        'featurize_fn': 'handcraft_vector', # what type of encoding to use

        # Learning Params ############################
        'EPS_START': 1.0,           # initial exploration rate (%)
        'EPS_END': 0.1,             # final exploration rate (%)
        'EPS_ERR': 0.01,            # EPS_ERR used to calculate EPS_DECAY
        'init_reward_shaping_scale': 1.0,  # initial reward shaping scale (decays to 0 with epsilon)
        'gamma': 0.95,              # discount factor
        'tau': 0.005,               # tau for soft update of target network
        "learning_rate": 1e-4,      # "learning_rate": 1e-2, "learning_rate": 1e-3, "learning_rate": 1e-5,
        "num_hidden_layers": 3,     # DDQN-MLP config
        "size_hidden_layers": 256,  # DDQN-MLP config
        "device": device,           # pytorch device
        "n_mini_batch": 1,          # number of mini batches to sample from replay memory
        "minibatch_size": 256,      # size of mini batch sampled from replay memory
        "replay_memory_size": 10_000, # size of replay memory

        # Evaluation Params ############################
        'test_interval': 10,        # how many iterations before testing again
        'N_tests': 1,               # number of tests to run (>1 if using test_rationality!='max')
        'test_rationality': 'max',  # how to choose actions during test time (max, #) where # is boltzmann rationality
        'logger_filter_size': 10,   # size of filter for logger
        'logger_update_period': 1,  # used to refresh the logger plot
    }


def main():
    # Parse config ----------------
    ALGORITHM = config['ALGORITHM']
    LAYOUT = config['LAYOUT']
    HORIZON = config['HORIZON']
    ITERATIONS = config['ITERATIONS']
    N_tests = config['N_tests']
    test_interval = config['test_interval']
    init_reward_shaping_scale = config['init_reward_shaping_scale']
    EPS_END = config['EPS_END']
    EPS_START = config['EPS_START']
    EPS_ERR = config['EPS_ERR']
    n_mini_batch = config['n_mini_batch']
    minibatch_size = config['minibatch_size']
    perc_random_start = config['perc_random_start']
    GAMMA = config['gamma']
    LR = config['learning_rate']
    TAU = config['tau']
    replay_memory_size = config['replay_memory_size']
    n_players = config['n_players']
    test_rationality = config['test_rationality']

    # Logger ----------------
    set_recorder(reward=FilteredLinePlot(filter_size=config['logger_filter_size'],
                                         xlabel="Iteration",  ylabel=f"Score ({LAYOUT}|{ALGORITHM})"))
    set_recorder(shaped_reward=FilteredLinePlot(filter_size=config['logger_filter_size'],
                                                xlabel="Iteration",  ylabel=f"Shaped Score ({LAYOUT}|{ALGORITHM})"))
    set_update_period(config['logger_update_period'])  # [seconds]

    # Generate MDP and environment----------------
    mdp = OvercookedGridworld.from_layout_name(LAYOUT)
    env = OvercookedEnv.from_mdp(mdp, horizon=HORIZON)
    state_manager = StartStateManager(mdp) # used for randomized start states

    # Generate Deep Models -------------
    obs_shape = mdp.get_lossless_vector_encoding_shape(n_players=n_players); config['obs_shape'] = obs_shape
    policy_net = DQN_vector_feature(obs_shape, config['n_actions'], size_hidden_layers=config['size_hidden_layers']).to(device)
    target_net = DQN_vector_feature(obs_shape, config['n_actions'], size_hidden_layers=config['size_hidden_layers']).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    replay_memory = ReplayMemory(replay_memory_size)

    # Generate agents ----------------
    q_agent = SoloDeepQAgent(mdp=mdp, agent_index=0, config=config, target_net=target_net, policy_net=policy_net)
    stay_agent = StayAgent()

    ######################################################################################
    # MAIN LOOP ##########################################################################
    ######################################################################################
    steps_done = 0
    iter_rewards = []
    for iter in range(ITERATIONS):
        # Calc Decay Parameters ----------------
        EPS_DECAY = int((-1. * ITERATIONS)/np.log(EPS_ERR))
        epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        r_shape_scale = (init_reward_shaping_scale) * math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1

        # reinitialize state ----------------
        env.reset()
        is_random_start = iter/ITERATIONS < perc_random_start
        env.state = state_manager.assign(env.state,
                                         random_loc=is_random_start,
                                         random_pot=is_random_start,
                                         random_held=is_random_start)

        ##############################################
        # Simulate Episode ----------------
        ##############################################
        cum_reward = 0
        shaped_reward = 0
        for t in count():
            # Take actions and form observations
            state = env.state
            obs = mdp.get_lossless_vector_encoding(state, n_players=n_players)
            joint_action,joint_action_info = q_agent.action(obs, epsilon = epsilon)
            # action1, action_info1 = q_agent.action(obs, epsilon = epsilon)
            # action2, _ = stay_agent.action(state)
            # joint_action = (action1, action2)
            next_state, reward, done, info = env.step(joint_action)

            if done:
                next_obs = None
            else:
                next_obs = mdp.get_lossless_vector_encoding(state, n_players=n_players)
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

            # Push experience to replay memory
            replay_memory.push(torch.tensor(obs, dtype=torch.float32, device=q_agent.device).unsqueeze(0),
                               # torch.tensor([[action_info1['action_index']]], dtype=torch.int64, device=device),
                               torch.tensor([[joint_action_info['action_index']]], dtype=torch.int64, device=device),
                               next_obs,
                               torch.tensor([reward + shaped_reward], device=device))
            # Log rewards
            shaped_reward += r_shape_scale * np.sum(info["shaped_r_by_agent"])/2
            # shaped_reward += r_shape_scale*info["shaped_r_by_agent"][0]
            cum_reward += reward

            # Optimize model
            if len(replay_memory) > 0.25*minibatch_size:
                for _ in range(n_mini_batch):
                    optimize_model(policy_net,target_net,optimizer,replay_memory,minibatch_size,GAMMA)

            # Soft update of the target network 's weights
            target_net = soft_update(policy_net, target_net, TAU)

            if done: break
            env.state = next_state

        # Report Episode ----------------
        print(f"Iteration {iter} "
              f"| train reward: {round(cum_reward,3)} "
              f"| shaped reward: {round(shaped_reward,3)} "
              f"| reward shaping scale {round(r_shape_scale,3)} "
              f"| memory len {len(replay_memory)} "
              # f"| Explore Rationality {exp_rationality} "
              f"| Explore Prob {epsilon} "
              )

        ##############################################
        # Test policy ---------------------
        ##############################################
        if iter % test_interval == 0:
            if debug: print('Test policy')
            test_reward = 0
            test_shaped_reward = 0
            for test in range(N_tests):
                env.reset()
                for t in count():
                    if debug: print(f'Test policy: test {test}, t {t}')
                    state = env.state
                    obs = mdp.get_lossless_vector_encoding(state, n_players=n_players)
                    joint_action, joint_action_info = q_agent.action(obs, epsilon=epsilon)
                    # action1, action_info1 = q_agent.action(obs, epsilon = epsilon)
                    # action2, _ = stay_agent.action(state)
                    # joint_action = (action1, action2)
                    next_state, reward, done, info = env.step(joint_action)
                    test_reward += reward
                    test_shaped_reward +=  info["shaped_r_by_agent"][0]
                    if done: break
                    env.state = next_state
            record(reward=test_reward / N_tests)
            record(shaped_reward=test_shaped_reward / N_tests)
            print(f"\nTest: | nTests= {N_tests} | Ave Reward = {test_reward / N_tests} | Ave Shaped Reward = {test_shaped_reward / N_tests}\n")
            # ftimer.report()
        # -------------------------------
    fig,ax = plt.subplots()
    ax.plot(iter_rewards)
    plt.show()


if __name__ == "__main__":
    main()