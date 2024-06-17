import numpy as np
from risky_overcooked_py.mdp.actions import Action, Direction
from risky_overcooked_py.agents.benchmarking import AgentEvaluator,LayoutGenerator
from risky_overcooked_py.agents.agent import Agent, AgentPair,StayAgent, RandomAgent, GreedyHumanModel
from risky_overcooked_rl.utils.custom_deep_agents import SoloDeepQAgent
# from risky_overcooked_rl.utils.deep_models import ReplayMemory,DQN_vector_feature
from risky_overcooked_rl.utils.deep_models_pytorch import ReplayMemory,DQN_vector_feature,device,optimize_model,soft_update#,select_action
from risky_overcooked_rl.utils.rl_logger import RLLogger
from risky_overcooked_rl.utils.rl_logger import FunctionTimer
# from risky_overcooked_py.mdp.overcooked_mdp import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
import pickle
import datetime
import os
from tempfile import TemporaryFile
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld,OvercookedState,SoupState, ObjectState
from itertools import product,count
# import matplotlib.pyplot as plt
import warnings
# from develocorder import (
#     LinePlot,
#     Heatmap,
#     FilteredLinePlot,
#     DownsampledLinePlot,
#     set_recorder,
#     record,
#     set_update_period,
#     set_num_columns,
# )
import random
import torch
import torch.nn as nn
import torch.optim as optim
import math

debug = False
config = {
        'ALGORITHM': 'solo_dqn_vector_egreedy',
        "seed": 41,

        # Env Params
        'LAYOUT': "cramped_room_single", 'HORIZON': 200, 'ITERATIONS': 3_000,
        # 'LAYOUT': "sanity_check3_single", 'HORIZON': 200, 'ITERATIONS': 500,
        # 'LAYOUT': "sanity_check4_single", 'HORIZON': 300, 'ITERATIONS': 5_000,
        # 'LAYOUT': "sanity_check3",
        # 'LAYOUT': "sanity_check2",
        'init_reward_shaping_scale': 1.0,
        "obs_shape": None, #TODO: how was this computing with incorrect dimensions? [18,5,4]
        "n_actions": 6,
        "perc_random_start": 0.01,
        'done_after_delivery': False,
        'start_with_soup': False,
        'featurize_fn': 'handcraft_vector',

        # Learning Params
        'max_exploration_proba': 1.0,
        'min_exploration_proba': 0.1,
        'gamma': 0.95,
        'tau': 0.005,# 0.005,# soft update of target network
        # "learning_rate": 1e-2,
        # "learning_rate": 1e-3,
        "learning_rate": 1e-4,
        # "learning_rate": 5e-5,
        # "learning_rate": 1e-5,
        'rationality': 'max', #  base rationality if not specified, ='max' for argmax
        'exp_rationality_range': [10,10],  # exploring rationality
        "num_filters": 25,  # CNN params
        "num_convs": 3,  # CNN params
        "num_hidden_layers": 3,  # MLP params
        "size_hidden_layers": 256,#32,  # MLP params
        "device": device,


        # "n_mini_batch": 1,
        "n_mini_batch": 1,
        "minibatch_size": 256,
        "replay_memory_size": 15_000,

        # Evaluation Params
        'test_interval': 10,
        'N_tests': 1,
        'test_rationality': 'max',  # rationality for explotation during testing
        'train_rationality': 'max', # rationality for explotation during training

        'logger_filter_size': 10,
        'logger_update_period': 1,  # [seconds]
    }

def random_start_state(mdp,rnd_obj_prob_thresh=0.25):
    # Random position
    random_state = mdp.get_random_start_state_fn(random_start_pos=True, rnd_obj_prob_thresh=0.0)()
    # random_state.players[0].position = mdp.start_player_positions[1]
    # If there are two players, make sure no overlapp
    # while np.all(np.array(random_state.players[1].position) == np.array(random_state.players[0].position)):
    #     random_state = mdp.get_random_start_state_fn(random_start_pos=True, rnd_obj_prob_thresh=0.0)()
    #     random_state.players[1].position = mdp.start_player_positions[1]
    # env.state = random_state

    # Arbitrary hard-coding for randomization of objects
    # For each pot, add a random amount of onions and tomatoes with prob rnd_obj_prob_thresh
    # Begin the soup cooking with probability rnd_obj_prob_thresh
    pots = mdp.get_pot_states(random_state)["empty"]
    for pot_loc in pots:
        p = np.random.rand()
        if p < rnd_obj_prob_thresh:
            n = int(np.random.randint(low=1, high=3))
            q = np.random.rand()
            # cooking_tick = np.random.randint(0, 20) if n == 3 else -1
            cooking_tick = 0 if n == 3 else -1
            random_state.objects[pot_loc] = SoupState.get_soup(
                pot_loc,
                num_onions=n,
                num_tomatoes=0,
                cooking_tick=cooking_tick,
            )

    # For each player, add a random object with prob rnd_obj_prob_thresh
    for player in random_state.players:
        p = np.random.rand()
        if p < rnd_obj_prob_thresh:
            # Different objects have different probabilities
            obj = np.random.choice(
                ["dish", "onion", "soup"], p=[0.2, 0.6, 0.2]
            )
            n = int(np.random.randint(low=1, high=4))
            if obj == "soup":
                player.set_object(
                    SoupState.get_soup(
                        player.position,
                        num_onions=n,
                        num_tomatoes=0,
                        finished=True,
                    )
                )
            else:
                player.set_object(ObjectState(obj, player.position))
    # random_state = mdp.get_random_start_state_fn(random_start_pos=True, rnd_obj_prob_thresh=0.25)()
    return random_state



def main():
    ALGORITHM = config['ALGORITHM']
    LAYOUT = config['LAYOUT']
    HORIZON = config['HORIZON']
    ITERATIONS = config['ITERATIONS']
    N_tests = config['N_tests']
    test_interval = config['test_interval']
    init_reward_shaping_scale = config['init_reward_shaping_scale']
    min_explore = config['min_exploration_proba']
    max_explore = config['max_exploration_proba']
    EPS_END = config['min_exploration_proba']
    EPS_START = config['max_exploration_proba']
    n_mini_batch = config['n_mini_batch']
    minibatch_size = config['minibatch_size']
    exp_rationality_range = config['exp_rationality_range']
    perc_random_start = config['perc_random_start']
    GAMMA = config['gamma']
    LR = config['learning_rate']
    TAU = config['tau']
    EPS_DECAY = 1000
    replay_memory_size = config['replay_memory_size']
    done_after_delivery = config['done_after_delivery']
    start_with_soup = config['start_with_soup']
    test_rationality = config['test_rationality']
    train_rationality = config['train_rationality']



    if start_with_soup:
        warnings.warn("Starting with soup")

    # Logger ----------------
    logger = RLLogger(rows = 2,cols = 1)
    logger.add_lineplot('test_reward',xlabel='iter',ylabel='$R_{test}$',filter_window=10,display_raw=True, loc = (0,1))
    logger.add_lineplot('train_reward', xlabel='iter', ylabel='$R_{train}$', filter_window=10, display_raw=True, loc=(1,1))
    # logger.add_lineplot('shaped_reward', xlabel='iter', ylabel='$R_{shape}$', filter_window=10, display_raw=True,  loc=(2, 1))
    logger.add_table('Params',config)

    # Generate MDP and environment----------------
    mdp = OvercookedGridworld.from_layout_name(LAYOUT)
    env = OvercookedEnv.from_mdp(mdp, horizon=HORIZON)
    # model = DQN(**config)
    replay_memory = ReplayMemory(replay_memory_size)


    # Initialize policy and target networks
    obs_shape = mdp.get_lossless_encoding_vector_shape(); config['obs_shape'] = obs_shape
    policy_net = DQN_vector_feature(obs_shape, config['n_actions'],size_hidden_layers=config['size_hidden_layers']).to(device)
    target_net = DQN_vector_feature(obs_shape, config['n_actions'],size_hidden_layers=config['size_hidden_layers']).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)


    # Generate agents
    # q_agent = SoloDeepQAgent(mdp,agent_index=0,model=DQN_vector_feature, config=config)
    q_agent = SoloDeepQAgent(mdp,agent_index=0,policy_net=policy_net, config=config)
    # obs_shape = q_agent.get_featurized_shape()
    stay_agent = StayAgent()


    steps_done = 0
    iter_rewards = []
    train_rewards = []
    for iter in range(ITERATIONS):
        logger.spin()
        # ftimer.clear()
        # r_shape_scale = init_reward_shaping_scale-(iter/ITERATIONS)*(init_reward_shaping_scale-0)

        # Boltzmann exploration
        # exp_rationality = exp_rationality_range[0] + (iter / ITERATIONS) * (exp_rationality_range[1] - exp_rationality_range[0])

        # e-greedy exploration
        # exploration_proba = max_explore-(iter/ITERATIONS)*(max_explore-min_explore)
        # exp_rationality = 'random' if np.random.uniform(0, 1)  < exploration_proba else config['rationality']
        EPS_START = max_explore
        EPS_END = min_explore
        EPS_ERR = 0.01
        EPS_DECAY = int((-1. * ITERATIONS)/np.log(EPS_ERR))
        exploration_proba = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)

        # r_shape_scale = init_reward_shaping_scale - (iter / ITERATIONS) * (init_reward_shaping_scale - 0)
        r_shape_scale = (init_reward_shaping_scale) * math.exp(-1. * steps_done / EPS_DECAY)



        steps_done += 1

        # Initialize the environment and state
        env.reset()
        if start_with_soup:
            player = env.state.players[0]
            soup = SoupState.get_soup(
                                player.position,
                                num_onions=3,
                                num_tomatoes=0,
                                finished=True,
                            )
            player.set_object(soup)
        elif iter/ITERATIONS < perc_random_start:
            env.state = random_start_state(mdp)


            # env.state.players[0].set_object(soup)

        # Simulate Episode ----------------
        cum_reward = 0
        shaped_reward = 0

        for t in count():
            state = env.state
            # obs = q_agent.featurize(state)
            # obs = mdp.get_lossless_encoding_vector(state)
            # obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            # ai1 = select_action(obs,policy_net,exploration_proba)#.numpy().flatten()[0]
            # action1 = Action.INDEX_TO_ACTION[ai1.numpy().flatten()[0]]
            # action1, _ = q_agent.action(state,rationality=exp_rationality)
            action1, action_info1 = q_agent.action(state,exp_prob=exploration_proba,rationality=train_rationality)
            action2, _ = stay_agent.action(state)
            joint_action = (action1, action2)
            next_state, reward, done, info = env.step(joint_action) # what is joint-action info?

            if reward==20 and done_after_delivery==True:
                done=True

            shaped_reward += r_shape_scale*info["shaped_r_by_agent"][0]
            cum_reward += reward


            if done: next_obs = None
            else: next_obs = q_agent.featurize(next_state)
                # next_obs = q_agent.featurize(next_state)
                # next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

            replay_memory.push(q_agent.featurize(state),
                               torch.tensor([action_info1['action_index']], dtype=torch.int64, device=device).unsqueeze(0),
                               next_obs,
                               torch.tensor([reward + shaped_reward], device=device))

            # replay_memory.push(obs,
            #                    ai1,#torch.tensor([ai1], dtype=torch.int64, device=device),
            #                    next_obs,
            #                    torch.tensor([reward + shaped_reward],device=device))
            if len(replay_memory) > 0.25*minibatch_size:
                for _ in range(n_mini_batch):
                    optimize_model(policy_net,target_net,optimizer,replay_memory,minibatch_size,GAMMA)

            #Soft update of the target network 's weights
            # θ′ ← τ θ + (1 −τ )θ′
            # target_net_state_dict = target_net.state_dict()
            # policy_net_state_dict = policy_net.state_dict()
            # for key in policy_net_state_dict:
            #     target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            # target_net.load_state_dict(target_net_state_dict)
            target_net = soft_update(policy_net,target_net,TAU)

            # if len(replay_memory) > 0.5*config["replay_memory_size"]:
            #     experiences = replay_memory.sample(minibatch_size)
            #     q_agent.update(experiences)
            # counters_are_full = (len(mdp.get_reachable_counters())-1==len(mdp.get_counter_objects_dict(state).keys()))
            # if counters_are_full:
            #     print(f'\n\nCOUNTER FULL!!!#########################\n')
            #     break


            if done:  break
            env.state = next_state


        train_rewards.append(cum_reward+shaped_reward)
        print(f"Iteration {iter} "
              f"| train reward: {round(cum_reward,3)} "
              f"| shaped reward: {round(shaped_reward,3)} "
              f"| reward shaping scale {round(r_shape_scale,3)} "
              f"| memory len {len(replay_memory)} "
              # f"| Explore Rationality {exp_rationality} "
              f"| Explore Prob {exploration_proba} "
              )
        # print(f"Iteration {iter} | cumulative train reward: {cum_reward} |  P(explore) {q_agent.exploration_proba} ")


        ##############################################
        # Test policy ---------------------
        ##############################################
        if iter % test_interval == 0:
            if debug: print('Test policy')
            test_reward = 0
            test_shaped_reward = 0
            for test in range(N_tests):
                env.reset()
                # env.state = mdp.get_standard_start_state()

                if start_with_soup:
                    player = env.state.players[0]
                    soup = SoupState.get_soup(
                        player.position,
                        num_onions=3,
                        num_tomatoes=0,
                        finished=True,
                    )
                    player.set_object(soup)

                for t in count():
                    if debug: print(f'Test policy: test {test}, t {t}')
                    state = env.state
                    # obs = q_agent.featurize(state)
                    # obs = mdp.get_lossless_encoding_vector(state)
                    # obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    # ai1 = select_action(obs, policy_net, 0,rationality=config['test_rationality'])  # .numpy().flatten()[0]
                    # action1 = Action.INDEX_TO_ACTION[ai1.numpy().flatten()[0]]
                    # # state = env.state
                    # # action1, _ = q_agent.action(state,rationality=config['test_rationality'])
                    action1, action_info1 = q_agent.action(state,exp_prob=0,rationality=test_rationality)

                    action2, _ = stay_agent.action(state)
                    joint_action = (action1, action2)
                    next_state, reward, done, info = env.step(joint_action)
                    test_reward += reward
                    test_shaped_reward +=  info["shaped_r_by_agent"][0]
                    if done: break

                    # env.state = next_state
            logger.log(test_reward=[iter, test_reward / N_tests], train_reward=[iter, np.mean(train_rewards)])
            logger.draw()
            # record(train_reward=np.mean(train_rewards),reward=test_reward / N_tests)#,shaped_reward=test_shaped_reward / N_tests
            # record(reward=test_reward / N_tests)
            # record(shaped_reward=test_shaped_reward / N_tests)
            print(f"\nTest: | nTests= {N_tests} | Ave Reward = {test_reward / N_tests} | Ave Shaped Reward = {test_shaped_reward / N_tests}\n")
            train_rewards = []
            # ftimer.report()
        # -------------------------------




    # fig,ax = plt.subplots()
    # ax.plot(iter_rewards)
    # plt.show()


if __name__ == "__main__":
    main()