import numpy as np
from risky_overcooked_py.mdp.actions import Action, Direction
from risky_overcooked_py.agents.benchmarking import AgentEvaluator,LayoutGenerator
from risky_overcooked_py.agents.agent import Agent, AgentPair,StayAgent, RandomAgent, GreedyHumanModel
from risky_overcooked_rl.utils.custom_deep_agents import SoloDeepQAgent
from risky_overcooked_rl.utils.deep_models import ReplayMemory,DQN_vector_feature
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
debug = False
config = {
        'ALGORITHM': 'solo_dqn_vector_egreedy',
        "seed": 41,

        # Env Params
        # 'LAYOUT': "cramped_room_single", 'HORIZON': 200, 'ITERATIONS': 10_000,
        'LAYOUT': "sanity_check3_single", 'HORIZON': 300, 'ITERATIONS': 1_000,
        # 'LAYOUT': "sanity_check4_single", 'HORIZON': 300, 'ITERATIONS': 5_000,
        # 'LAYOUT': "sanity_check3",
        # 'LAYOUT': "sanity_check2",
        'init_reward_shaping_scale': 1.0,
        "obs_shape": None, #TODO: how was this computing with incorrect dimensions? [18,5,4]
        "n_actions": 6,
        "perc_random_start": 0.05,
        'done_after_delivery': True,
        'start_with_soup': False,

        # Learning Params
        'max_exploration_proba': 0.90,
        'min_exploration_proba': 0.05,
        'gamma': 0.95,
        # "learning_rate": 1e-2,
        "learning_rate": 1e-3,
        # "learning_rate": 1e-4,
        'rationality': 10,#'max', #  base rationality if not specified, ='max' for argmax
        'exp_rationality_range': [10,10],  # exploring rationality
        "num_filters": 25,  # CNN params
        "num_convs": 3,  # CNN params
        "num_hidden_layers": 3,  # MLP params
        "size_hidden_layers": 32,  # MLP params


        # "n_mini_batch": 1,
        "n_mini_batch": 10,
        "minibatch_size": 32,
        "replay_memory_size": 5_000,

        # Evaluation Params
        'test_interval': 10,
        'N_tests': 1,
        'test_rationality': 'max',
        'logger_filter_size': 10,
        'logger_update_period': 5,  # [seconds]
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
    exp_rationality_range = config['exp_rationality_range']
    perc_random_start = config['perc_random_start']
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
    replay_memory = ReplayMemory(config["replay_memory_size"])

    # Generate agents
    q_agent = SoloDeepQAgent(mdp,agent_index=0,model=DQN_vector_feature,featurize_fn='handcraft_vector', config=config)
    q_agent.get_featurized_shape()
    stay_agent = StayAgent()


    iter_rewards = []
    for iter in range(ITERATIONS):
        # ftimer.clear()
        r_shape_scale = init_reward_shaping_scale-(iter/ITERATIONS)*(init_reward_shaping_scale-0)

        # Boltzmann exploration
        # exp_rationality = exp_rationality_range[0] + (iter / ITERATIONS) * (exp_rationality_range[1] - exp_rationality_range[0])

        # e-greedy exploration
        exploration_proba = max_explore-(iter/ITERATIONS)*(max_explore-min_explore)
        exp_rationality = 'random' if np.random.uniform(0, 1)  < exploration_proba else config['rationality']


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
        # Random start state
        elif iter/ITERATIONS < perc_random_start:
            env.state = random_start_state(mdp)


            # env.state.players[0].set_object(soup)
        # Simulate Episode ----------------
        cum_reward = 0
        shaped_reward = 0

        exp_buffer = []
        for t in count():
            state = env.state
            action1, _ = q_agent.action(state,rationality=exp_rationality)
            action2, _ = stay_agent.action(state)
            joint_action = (action1, action2)
            next_state, reward, done, info = env.step(joint_action) # what is joint-action info?

            if reward==20 and done_after_delivery==True:
                done=True

            shaped_reward += r_shape_scale*info["shaped_r_by_agent"][0]
            cum_reward += reward

            exp_buffer.append([q_agent.featurize(state),
                               Action.ACTION_TO_INDEX[action1],
                               reward + shaped_reward,
                               q_agent.featurize(next_state),
                               done])
            # replay_memory.push(q_agent.featurize(state),
            #                    Action.ACTION_TO_INDEX[action1],
            #                    reward + shaped_reward,
            #                    q_agent.featurize(next_state),
            #                    done)

            # if len(replay_memory) > 0.5*config["replay_memory_size"]:
            #     experiences = replay_memory.sample(minibatch_size)
            #     q_agent.update(experiences)
            # counters_are_full = (len(mdp.get_reachable_counters())-1==len(mdp.get_counter_objects_dict(state).keys()))
            # if counters_are_full:
            #     print(f'\n\nCOUNTER FULL!!!#########################\n')
            #     break


            if done:  break
            env.state = next_state

        for t,exp in enumerate(exp_buffer):

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
                    action1, _ = q_agent.action(state,rationality=config['test_rationality'])

                    if t<2:
                        # obs = q_agent.featurize(env.state)
                        obs = q_agent.featurize(state)
                        qvals = q_agent.model.predict(obs[np.newaxis]).flatten()
                        a_probs = q_agent.softmax(qvals)
                        action_probs_str = f'[{t}] A={Action.to_char(action1)}\t' + ''.join(
                            [f'\t | \t{Action.to_char(Action.INDEX_TO_ACTION[i])}: {np.round(p, 3)}' for i, p in
                             enumerate(a_probs)])
                        print(f"T{test} A-probs: {action_probs_str}")

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

                    # env.state = next_state
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