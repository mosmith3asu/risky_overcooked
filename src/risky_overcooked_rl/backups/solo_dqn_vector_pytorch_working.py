import numpy as np
from risky_overcooked_py.mdp.actions import Action, Direction
from risky_overcooked_py.agents.benchmarking import AgentEvaluator,LayoutGenerator
from risky_overcooked_py.agents.agent import Agent, AgentPair,StayAgent, RandomAgent, GreedyHumanModel
from risky_overcooked_rl.utils.custom_deep_agents import SoloDeepQAgent
# from risky_overcooked_rl.utils.deep_models import ReplayMemory,DQN_vector_feature
from risky_overcooked_rl.utils.deep_models_pytorch import ReplayMemory,DQN_vector_feature,Transition,device

from risky_overcooked_rl.utils.rl_logger import FunctionTimer
# from risky_overcooked_py.mdp.overcooked_mdp import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
import pickle
import datetime
import os
from tempfile import TemporaryFile
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld,OvercookedState,SoupState, ObjectState
from itertools import product,count
import matplotlib.pyplot as plt
import warnings
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
        # 'LAYOUT': "cramped_room_single", 'HORIZON': 200, 'ITERATIONS': 10_000,
        'LAYOUT': "sanity_check3_single", 'HORIZON': 400, 'ITERATIONS': 500,
        # 'LAYOUT': "sanity_check4_single", 'HORIZON': 300, 'ITERATIONS': 5_000,
        # 'LAYOUT': "sanity_check3",
        # 'LAYOUT': "sanity_check2",
        'init_reward_shaping_scale': 1.0,
        "obs_shape": None, #TODO: how was this computing with incorrect dimensions? [18,5,4]
        "n_actions": 6,
        "perc_random_start": 0.00,
        'done_after_delivery': False,
        'start_with_soup': False,

        # Learning Params
        'max_exploration_proba': 0.9,
        'min_exploration_proba': 0.1,
        'gamma': 0.95,
        # "learning_rate": 1e-2,
        # "learning_rate": 1e-3,
        "learning_rate": 1e-4,
        # "learning_rate": 1e-5,
        'rationality': 10,#'max', #  base rationality if not specified, ='max' for argmax
        'exp_rationality_range': [10,10],  # exploring rationality
        "num_filters": 25,  # CNN params
        "num_convs": 3,  # CNN params
        "num_hidden_layers": 3,  # MLP params
        "size_hidden_layers": 256,#32,  # MLP params
        "device": device,


        # "n_mini_batch": 1,
        "n_mini_batch": 1,
        "minibatch_size": 256,
        "replay_memory_size": 8_000,

        # Evaluation Params
        'test_interval': 10,
        'N_tests': 1,
        'test_rationality': 'max',
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
            obj = np.random.choice( ["dish", "onion", "soup"], p=[0.2, 0.6, 0.2])
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


def select_action(state,policy_net,exp_prob,n_actions=6,debug=False):
    # global steps_done
    sample = random.random()
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #     math.exp(-1. * steps_done / EPS_DECAY)
    # steps_done += 1
    if sample < exp_prob:
        # return np.random.choice(np.arange(n_actions))
        action = Action.sample(np.ones(n_actions)/n_actions)
        ai = Action.ACTION_TO_INDEX[action]
        return torch.tensor([[ai]], device=device, dtype=torch.long)
        # return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    else:
        if debug: print('Greedy')
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # return policy_net(state).max(1).indices.view(1, 1).numpy().flatten()[0]
            return policy_net(state).max(1).indices.view(1, 1)



def optimize_model(policy_net,target_net,optimizer,memory,BATCH_SIZE,GAMMA):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)#.unsqueeze(0)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


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
    TAU = 0.005
    EPS_DECAY = 1000
    replay_memory_size = config['replay_memory_size']
    done_after_delivery = config['done_after_delivery']
    start_with_soup = config['start_with_soup']



    if start_with_soup:
        warnings.warn("Starting with soup")

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
    # model = DQN(**config)
    replay_memory = ReplayMemory(replay_memory_size)

    # Generate agents
    q_agent = SoloDeepQAgent(mdp,agent_index=0,model=DQN_vector_feature,featurize_fn='handcraft_vector', config=config)
    obs_shape = q_agent.get_featurized_shape()
    stay_agent = StayAgent()

    policy_net = DQN_vector_feature(obs_shape, config['n_actions'],size_hidden_layers=config['size_hidden_layers']).to(device)
    target_net = DQN_vector_feature(obs_shape, config['n_actions'],size_hidden_layers=config['size_hidden_layers']).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

    steps_done = 0
    iter_rewards = []
    for iter in range(ITERATIONS):
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
            obs = q_agent.featurize(state)
            obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            ai1 = select_action(obs,policy_net,exploration_proba)#.numpy().flatten()[0]
            action1 = Action.INDEX_TO_ACTION[ai1.numpy().flatten()[0]]
            # action1, _ = q_agent.action(state,rationality=exp_rationality)
            action2, _ = stay_agent.action(state)
            joint_action = (action1, action2)
            next_state, reward, done, info = env.step(joint_action) # what is joint-action info?

            if reward==20 and done_after_delivery==True:
                done=True

            shaped_reward += r_shape_scale*info["shaped_r_by_agent"][0]
            cum_reward += reward

            if done:
                next_obs = None
            else:
                next_obs = q_agent.featurize(next_state)
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

            replay_memory.push(obs,
                               ai1,#torch.tensor([ai1], dtype=torch.int64, device=device),
                               next_obs,
                               torch.tensor([reward + shaped_reward],device=device))
            if len(replay_memory) > 0.25*minibatch_size:
                for _ in range(n_mini_batch):
                    optimize_model(policy_net,target_net,optimizer,replay_memory,minibatch_size,GAMMA)

            #Soft update of the target network 's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            # if len(replay_memory) > 0.5*config["replay_memory_size"]:
            #     experiences = replay_memory.sample(minibatch_size)
            #     q_agent.update(experiences)
            # counters_are_full = (len(mdp.get_reachable_counters())-1==len(mdp.get_counter_objects_dict(state).keys()))
            # if counters_are_full:
            #     print(f'\n\nCOUNTER FULL!!!#########################\n')
            #     break


            if done:  break
            env.state = next_state



        print(f"Iteration {iter} "
              f"| train reward: {round(cum_reward,3)} "
              f"| shaped reward: {round(shaped_reward,3)} "
              f"| reward shaping scale {round(r_shape_scale,3)} "
              f"| memory len {len(replay_memory)} "
              # f"| Explore Rationality {exp_rationality} "
              f"| Explore Prob {exploration_proba} "
              )
        # print(f"Iteration {iter} | cumulative train reward: {cum_reward} |  P(explore) {q_agent.exploration_proba} ")

        # -------------------------------
        # for _ in range(n_mini_batch):
        #     # experiences = ftimer('sample',  replay_memory.sample, minibatch_size)
        #     # ftimer('update',q_agent.update,experiences)
        #     experiences = replay_memory.sample(minibatch_size)
        #     q_agent.update(experiences)
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
                    obs = q_agent.featurize(state)
                    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    ai1 = select_action(obs, policy_net, 0)  # .numpy().flatten()[0]
                    action1 = Action.INDEX_TO_ACTION[ai1.numpy().flatten()[0]]
                    # state = env.state
                    # action1, _ = q_agent.action(state,rationality=config['test_rationality'])

                    action2, _ = stay_agent.action(state)
                    joint_action = (action1, action2)
                    next_state, reward, done, info = env.step(joint_action)
                    test_reward += reward
                    test_shaped_reward +=  info["shaped_r_by_agent"][0]
                    if done: break

                    # env.state = next_state
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