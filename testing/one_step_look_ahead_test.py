import numpy as np
from risky_overcooked_py.agents.agent import Agent, AgentPair,StayAgent, RandomAgent, GreedyHumanModel
from risky_overcooked_rl.utils.custom_deep_agents import SoloDeepQAgent,SelfPlay_DeepAgentPair
from risky_overcooked_rl.utils.deep_models import ReplayMemory_CPT,DQN_vector_feature,device,optimize_model_td_targets,soft_update
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
        'ALGORITHM': 'CLDE_DDQN',
        'Date': datetime.now().strftime("%m/%d/%Y, %H:%M"),

        # Env Params ----------------
        # 'LAYOUT': "risky_cramped_room_CLCE", 'HORIZON': 200, 'ITERATIONS': 5_000,
        'LAYOUT': "cramped_room_CLCE", 'HORIZON': 200, 'ITERATIONS': 3_000,
        "obs_shape": None,                  # computed dynamically based on layout
        "n_actions": 36,                    # number of agent actions
        "perc_random_start": 0.01,          # percentage of ITERATIONS with random start states
        # "perc_random_start": 0.9,          # percentage of ITERATIONS with random start states
        "equalib_sol": "pareto",               # equilibrium solution for testing
        # "equalib_sol": "QRE5",               # equilibrium solution for testing
        # "equalib_sol": "NASH",               # equilibrium solution for testing

        # Learning Params ----------------
        'epsilon_range': [1.0,0.1],         # epsilon-greedy range (start,end)
        'gamma': 0.95,                      # discount factor
        'tau': 0.005,                       # soft update weight of target network
        "lr": 1e-4,                         # learning rate
        "num_hidden_layers": 3,             # MLP params
        "size_hidden_layers": 256,#32,      # MLP params
        "device": device,
        "n_mini_batch": 1,              # number of mini-batches per iteration
        "minibatch_size": 256,          # size of mini-batches
        "replay_memory_size": 10_000,   # size of replay memory

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



def main():
    # Parse Config ----------------
    ALGORITHM = config['ALGORITHM']
    LAYOUT = config['LAYOUT']
    HORIZON = config['HORIZON']
    ITERATIONS = config['ITERATIONS']
    EPS_START, EPS_END = config['epsilon_range']
    n_mini_batch = config['n_mini_batch']
    minibatch_size = config['minibatch_size']
    perc_random_start = config['perc_random_start']
    GAMMA = config['gamma']
    LR = config['lr']
    TAU = config['tau']
    replay_memory_size = config['replay_memory_size']
    test_rationality = config['test_rationality']
    equalib_sol = config['equalib_sol']
    train_rationality = config['train_rationality']
    init_reward_shaping_scale = 1                   # decaying reward shaping weight
    N_tests = 1 if test_rationality=='max' else 3   # number of tests (only need 1 with max rationality)
    test_interval = 10                              # test every n iterations


    # Generate MDP and environment----------------
    mdp = OvercookedGridworld.from_layout_name(LAYOUT)
    env = OvercookedEnv.from_mdp(mdp, horizon=HORIZON)
    replay_memory = ReplayMemory_CPT(replay_memory_size)

    # Initialize policy and target networks ----------------
    obs_shape = mdp.get_lossless_encoding_vector_shape(); config['obs_shape'] = obs_shape
    policy_net = DQN_vector_feature(obs_shape, config['n_actions'],size_hidden_layers=config['size_hidden_layers']).to(device)
    target_net = DQN_vector_feature(obs_shape, config['n_actions'],size_hidden_layers=config['size_hidden_layers']).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

    # Generate agents ----------------
    q_agent1 = SoloDeepQAgent(mdp,agent_index=0,policy_net=policy_net,target_net=target_net,optimizer=optimizer, config=config)
    q_agent2 = SoloDeepQAgent(mdp,agent_index=1,policy_net=policy_net,target_net=target_net,optimizer=optimizer, config=config)
    agent_pair = SelfPlay_DeepAgentPair(q_agent1,q_agent2,equalib=equalib_sol)

    # Initiate Logger ----------------
    traj_visualizer = TrajectoryVisualizer(env)
    logger = RLLogger(rows=2, cols=1, num_iterations=ITERATIONS)
    logger.add_lineplot('test_reward', xlabel='iter', ylabel='$R_{test}$', filter_window=10, display_raw=True, loc=(0, 1))
    logger.add_lineplot('train_reward', xlabel='iter', ylabel='$R_{train}$', filter_window=10, display_raw=True, loc=(1, 1))
    logger.add_table('Params', config)
    logger.add_status()
    logger.add_button('Preview Game', callback=traj_visualizer.preview_qued_trajectory)


    ##############################################
    # TRAIN LOOP #################################
    ##############################################
    steps_done = 0
    train_rewards = []
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
        if iter/ITERATIONS < perc_random_start: env.state = random_start_state(mdp)

        # Simulate Episode ----------------
        cum_reward = 0
        shaped_reward = np.zeros(2)

        obs = agent_pair.featurize(env.state)
        for t in count():

            joint_action, action_info = agent_pair.action(obs,exp_prob=exploration_proba)
            joint_action_idx = action_info['action_index']

            next_state, reward, done, info = env.step(joint_action)
            shaped_reward += r_shape_scale * np.array(info["shaped_r_by_agent"])
            cum_reward += reward
            if done: next_obs = None
            else: next_obs = agent_pair.featurize(next_state)

            # Calc TD-Target -----------------------------
            TD_Targets = agent_pair.one_step_ahead_td_target(env.state,reward,joint_action_idx)
            # expQ = np.zeros(2)
            # prospects = mdp.one_step_lookahead(env.state.deepcopy(),
            #                                    joint_action=Action.ALL_JOINT_ACTIONS[joint_action_idx])
            #
            # for p in prospects:
            #     P_st_prime = p[2]
            #     st_prime = p[1]
            #     _, next_action_info = agent_pair.action(st_prime, exp_prob=0,use_target_network=True)
            #     joint_action_prob = next_action_info['action_probs']
            #     joint_action_Q = next_action_info['joint_action_Q']
            #     vals_st_prime = np.sum(joint_action_prob * joint_action_Q, axis=1)
            #     expQ += P_st_prime * vals_st_prime
            # TD_Targets = agent_pair.cpt_valuation(reward + GAMMA * expQ)


            # Store the transition in memory (featurized tensors) ----------------
            replay_memory.push(obs,#q_agent1.featurize(state),
                               torch.tensor([joint_action_idx], dtype=torch.int64, device=device).unsqueeze(0),
                               torch.tensor(np.array([TD_Targets]), device=device))

            # Optimize Model ----------------
            if len(replay_memory) > minibatch_size:
                for _ in range(n_mini_batch):
                    transitions = replay_memory.sample(minibatch_size)
                    # for ip in range(2): # optimize for each player
                        # optimize_model_td_targets(policy_net, target_net, optimizer, transitions, GAMMA,player=ip)
                    agent_pair.update(transitions)
                    # optimize_model(policy_net,target_net,optimizer,replay_memory,minibatch_size,GAMMA)

            # Soft update of the target network  ----------------
            target_net = soft_update(policy_net,target_net,TAU)

            if done:  break
            obs = next_obs


        train_rewards.append(cum_reward+shaped_reward)
        print(f"Iteration {iter} "
              f"| train reward: {round(cum_reward,3)} "
              f"| shaped reward: {np.round(shaped_reward,3)} "
              f"| memory len {len(replay_memory)} "
              f"| reward shaping scale {round(r_shape_scale,3)} "
              f"| Explore Prob {exploration_proba} "
              )
        logger.end_iteration()

        ##############################################
        # Test policy ################################
        ##############################################
        if iter % test_interval == 0:
            if debug: print('Test policy')
            test_reward = 0
            test_shaped_reward = 0

            for test in range(N_tests):
                state_history = []
                env.reset()
                for t in count():
                    if debug: print(f'Test policy: test {test}, t {t}')
                    state = env.state
                    state_history.append(state.deepcopy())
                    joint_action, action_info = agent_pair.action(state, exp_prob=exploration_proba)
                    next_state, reward, done, info = env.step(joint_action)
                    test_reward += reward
                    test_shaped_reward +=  info["shaped_r_by_agent"][0]
                    if done: break

                    # env.state = next_state
                traj_visualizer.que_trajectory(state_history)
            logger.log(test_reward=[iter, test_reward / N_tests], train_reward=[iter, np.mean(train_rewards)])
            logger.draw()
            print(f"\nTest: | nTests= {N_tests} | Ave Reward = {test_reward / N_tests} | Ave Shaped Reward = {test_shaped_reward / N_tests}\n")
            train_rewards = []
        # -------------------------------

    # Halt Program Until Close Plot ----------------
    logger.wait_for_close(enable=True)


if __name__ == "__main__":
    main()