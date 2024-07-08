import numpy as np
from risky_overcooked_py.agents.agent import Agent, AgentPair,StayAgent, RandomAgent, GreedyHumanModel
from risky_overcooked_rl.utils.custom_deep_agents import SoloDeepQAgent,SelfPlay_DeepAgentPair
from risky_overcooked_rl.utils.deep_models import ReplayMemory,DQN_vector_feature,device,SelfPlay_QRE_OSA
from risky_overcooked_rl.utils.rl_logger import RLLogger,TrajectoryVisualizer
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld,OvercookedState,SoupState, ObjectState
from risky_overcooked_py.mdp.actions import Action
from itertools import product,count
import torch
import torch.optim as optim
import math
from datetime import datetime
debug = False
config = {
        'ALGORITHM': 'CLDE_QRE-DDQN-OSA',
        'Date': datetime.now().strftime("%m/%d/%Y, %H:%M"),

        # Env Params ----------------
        'LAYOUT': "risky_coordination_ring", 'HORIZON': 200, 'ITERATIONS': 15_000,
        # 'LAYOUT': "risky_cramped_room_CLCE", 'HORIZON': 200, 'ITERATIONS': 15_000,
        # 'LAYOUT': "cramped_room_CLCE", 'HORIZON': 200, 'ITERATIONS': 15_000,
        # 'LAYOUT': "super_cramped_room", 'HORIZON': 200, 'ITERATIONS': 10_000,

        "obs_shape": None,                  # computed dynamically based on layout
        "n_actions": 36,                    # number of agent actions
        "perc_random_start": 0.2,          # percentage of ITERATIONS with random start states
        # "perc_random_start": 0.9,          # percentage of ITERATIONS with random start states
        "equalib_sol": "QRE",               # equilibrium solution for testing

        # Learning Params ----------------
        'epsilon_range': [1.0,0.05],         # epsilon-greedy range (start,end)
        'gamma': 0.95,                      # discount factor
        'tau': 0.005,                       # soft update weight of target network
        # "lr": 1e-4,                         # learning rate
        "lr": 1e-4,                         # learning rate
        "num_hidden_layers": 4,             # MLP params
        "size_hidden_layers": 256,#32,      # MLP params
        "device": device,
        "n_mini_batch": 1,              # number of mini-batches per iteration
        "minibatch_size": 128,          # size of mini-batches
        "replay_memory_size": 20_000,   # size of replay memory

        # Evaluation Param ----------------
        'test_rationality': 'max',  # rationality for exploitation during testing
        'train_rationality': 'max', # rationality for exploitation during training
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


def get_random_start_state_fn(mdp):
    random_state = mdp.get_random_start_state_fn(random_start_pos=True, rnd_obj_prob_thresh=0.0)()
    return random_state
def add_rand_pot_state(mdp,state,rnd_obj_prob_thresh=0.5):
    pots = mdp.get_pot_states(state)["empty"]
    for pot_loc in pots:
        p = np.random.rand()
        if p < rnd_obj_prob_thresh:
            n = int(np.random.randint(low=1, high=3))
            q = np.random.rand()
            # cooking_tick = np.random.randint(0, 20) if n == 3 else -1
            cooking_tick = 0 if n == 3 else -1
            state.objects[pot_loc] = SoupState.get_soup(
                pot_loc,
                num_onions=n,
                num_tomatoes=0,
                cooking_tick=cooking_tick,
            )
    return state
def add_rand_object(state,prog,rnd_obj_prob_thresh=0.8):
    obj_probs = [[0.05, 0.05,0.9],
                 [0.1, 0.3,0.6],
                 [0.2, 0.4,0.4],
                 [0.3, 0.4,0.3],
                 [0.4, 0.4,0.2],
                 [0.5, 0.3,0.2],
                 [0.6, 0.2,0.2],
                 [0.6, 0.2, 0.2],
                 ]
    obj_prob = obj_probs[math.floor(prog*len(obj_probs))]


    # For each player, add a random object with prob rnd_obj_prob_thresh
    for player in state.players:
        p = np.random.rand()
        if p < rnd_obj_prob_thresh:
            # Different objects have different probabilities
            obj = np.random.choice(
                [ "onion","dish", "soup"], p=obj_prob
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
    return state


def main():
    for key,val in config.items():
        print(f'{key}={val}')
    # Parse Config ----------------
    LAYOUT = config['LAYOUT']
    HORIZON = config['HORIZON']
    ITERATIONS = config['ITERATIONS']
    EPS_START, EPS_END = config['epsilon_range']
    perc_random_start = config['perc_random_start']
    test_rationality = config['test_rationality']
    init_reward_shaping_scale = 2                   # decaying reward shaping weight
    N_tests = 1 if test_rationality=='max' else 3   # number of tests (only need 1 with max rationality)
    test_interval = 10                              # test every n iterations


    # Generate MDP and environment----------------
    mdp = OvercookedGridworld.from_layout_name(LAYOUT)
    env = OvercookedEnv.from_mdp(mdp, horizon=HORIZON)

    # Initialize policy and target networks ----------------
    obs_shape = mdp.get_lossless_encoding_vector_shape(); config['obs_shape'] = obs_shape
    test_net = SelfPlay_QRE_OSA(obs_shape, config['n_actions'],config)

    # Initiate Logger ----------------
    traj_visualizer = TrajectoryVisualizer(env)
    logger = RLLogger(rows=3, cols=1, num_iterations=ITERATIONS)
    logger.add_lineplot('test_reward', xlabel='', ylabel='$R_{test}$', filter_window=10, display_raw=True,  loc=(0, 1))
    logger.add_lineplot('train_reward', xlabel='', ylabel='$R_{train}$', filter_window=10, display_raw=True,   loc=(1, 1))
    logger.add_lineplot('loss', xlabel='iter', ylabel='$Loss$', filter_window=10, display_raw=True, loc=(2, 1))
    logger.add_table('Params', config)
    logger.add_status()
    logger.add_button('Preview Game', callback=traj_visualizer.preview_qued_trajectory)

    ##############################################
    # TRAIN LOOP #################################
    ##############################################
    steps_done = 0
    train_rewards = []
    losses = []
    for iter in range(ITERATIONS):
        logger.start_iteration()
        # Step Decaying Params ----------------
        logger.spin()
        DECAY = int((-1. * ITERATIONS)/np.log(0.01)) # decay to 1% error of ending value
        exploration_proba = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / DECAY)
        r_shape_scale = (init_reward_shaping_scale) * math.exp(-1. * steps_done / DECAY)
        steps_done += 1

        # Initialize the environment and state ----------------
        env.reset()
        if iter / ITERATIONS < perc_random_start:
            prog = iter / (perc_random_start * ITERATIONS)
            # env.state = random_start_state(mdp)
            state = get_random_start_state_fn(mdp)
            state = add_rand_pot_state(mdp,state)
            env.state = add_rand_object(state, prog)
        # Simulate Episode ----------------
        cum_reward = 0
        shaped_reward = np.zeros(2)
        state = env.state
        obs = torch.tensor(mdp.get_lossless_encoding_vector(state), dtype=torch.float32, device=device).unsqueeze(0)


        for t in count():
            # obs = torch.tensor(mdp.get_lossless_encoding_vector(state), dtype=torch.float32, device=device).unsqueeze(0)
            # joint_action,joint_action_idx,action_probs = test_net.choose_joint_action(obs,epsilon=exploration_proba)
            # prospects = mdp.one_step_lookahead(env.state.deepcopy(),  joint_action=Action.ALL_JOINT_ACTIONS[joint_action_idx], as_tensor=True,device=device)
            # next_state, reward, done, info = env.step(joint_action)
            # shaped_reward += r_shape_scale * np.array(info["shaped_r_by_agent"])
            # cum_reward += reward

            old_state = env.state.deepcopy()

            joint_action, joint_action_idx, action_probs = test_net.choose_joint_action(obs, epsilon=exploration_proba)
            next_state, reward, done, info = env.step(joint_action)
            shaped_reward += r_shape_scale * np.array(info["shaped_r_by_agent"])
            cum_reward += reward



            # Store in memory ----------------
            next_obs = torch.tensor(mdp.get_lossless_encoding_vector(next_state),
                                    dtype=torch.float32, device=device).unsqueeze(0)
            prospects = mdp.one_step_lookahead(old_state,
                                               joint_action=Action.ALL_JOINT_ACTIONS[joint_action_idx], as_tensor=True,
                                               device=device)
            # assert torch.all(prospects[0][1]==next_obs), "prospects[0][1] != next_obs"

            rewards = np.array([reward + shaped_reward]).flatten()
            test_net.memory_double_push(state=obs,
                                        action=joint_action_idx,
                                        rewards = rewards,
                                        next_prospects=prospects,
                                        done = done)
            # Update model ----------------
            loss = test_net.update()
            if loss is not None: losses.append(loss)
            if done:  break
            obs = next_obs


        train_rewards.append(cum_reward+shaped_reward)
        print(f"Iteration {iter} "
              f"| train reward: {round(cum_reward,3)} "
              f"| shaped reward: {np.round(shaped_reward,3)} "
              f"| memory len {test_net.memory_len} "
              f"| reward shaping scale {round(r_shape_scale,3)} "
              f"| Explore Prob {exploration_proba} "
              )

        logger.end_iteration()
        ##############################################
        # Test policy ################################
        ##############################################
        if iter % test_interval == 0 and iter > 2:
            if debug: print('Test policy')
            test_rewards = []
            test_shaped_rewards = []

            for test in range(N_tests):
                test_reward, test_shaped_reward, state_history, action_history,aprob_history = test_policy(env,test_net)
                test_rewards.append(test_reward)
                test_shaped_rewards.append(test_shaped_reward)
                traj_visualizer.que_trajectory(state_history)

            logger.log(test_reward=[iter, np.mean(test_rewards)],
                       train_reward=[iter, np.mean(train_rewards)],
                       loss=[iter, np.mean(losses)])
            logger.draw()
            print(f"\nTest: | nTests= {N_tests} "
                  f"| Ave Reward = { np.mean(test_rewards)} "
                  f"| Ave Shaped Reward = { np.mean(test_shaped_rewards)}"
                  # f"\n{action_history}\n"
                  # f"{aprob_history[0]}\n"
                  )
            train_rewards = []
            losses = []

        # -------------------------------

    # Halt Program Until Close Plot ----------------
    logger.wait_for_close(enable=True)

def test_policy(env,model):
    env.reset()
    mdp = env.mdp
    state = env.state
    obs = torch.tensor(mdp.get_lossless_encoding_vector(state), dtype=torch.float32, device=device).unsqueeze(0)

    test_reward = 0
    test_shaped_reward = 0
    state_history = [state.deepcopy()]
    action_history = []
    aprob_history = []

    for t in count():
        joint_action, joint_action_idx,action_probs = model.choose_joint_action(obs, epsilon=0)
        next_state, reward, done, info = env.step(joint_action)
        action_history.append(joint_action_idx)
        aprob_history.append(action_probs)
        state_history.append(next_state.deepcopy())
        next_obs = torch.tensor(mdp.get_lossless_encoding_vector(next_state), dtype=torch.float32, device=device).unsqueeze(0)
        test_reward += reward
        test_shaped_reward += info["shaped_r_by_agent"][0]
        if done:  break
        obs = next_obs
    return test_reward,test_shaped_reward,state_history,action_history,aprob_history


if __name__ == "__main__":
    main()