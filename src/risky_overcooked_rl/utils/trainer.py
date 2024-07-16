import numpy as np
from risky_overcooked_rl.utils.deep_models import ReplayMemory,DQN_vector_feature,device,SelfPlay_QRE_OSA,SelfPlay_QRE_OSA_CPT
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

class Trainer:
    def __init__(self,model_object,config):

        # Generate MDP and environment----------------
        config['AGENT'] = model_object.__name__
        LAYOUT = config['LAYOUT']
        HORIZON = config['HORIZON']
        self.shared_rew = False

        self.ITERATIONS = config['ITERATIONS']
        self.mdp = OvercookedGridworld.from_layout_name(LAYOUT)
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=HORIZON)
        self.perc_random_start = config['perc_random_start']
        self.N_tests = 1
        self.test_interval = 10  # test every n iterations

        # Define Parameter Schedules ------------------
        EPS_START, EPS_END, EPS_DUR = config['epsilon_sched']
        RAT_START, RAT_END, RAT_DUR = config['rationality_sched']
        RSHAPE_START, RSHAPE_END, RSHAPE_DUR = config['rshape_sched']
        self.test_rationality = config['test_rationality']
        self.rationality_sched = np.hstack( [np.linspace(RAT_START, RAT_END, RAT_DUR), RAT_END * np.ones(self.ITERATIONS - RAT_DUR)])
        self.epsilon_sched = np.hstack([np.linspace(EPS_START, EPS_END, EPS_DUR), EPS_END * np.ones(self.ITERATIONS - EPS_DUR)])
        self.rshape_sched = np.hstack([np.linspace(RSHAPE_START, RSHAPE_END, RSHAPE_DUR), RSHAPE_END * np.ones(self.ITERATIONS - RSHAPE_DUR)])

        # Initialize policy and target networks ----------------
        obs_shape = self.mdp.get_lossless_encoding_vector_shape()
        config['obs_shape'] = obs_shape
        n_actions = 36
        self.model = model_object(obs_shape, n_actions, config) #SelfPlay_QRE_OSA(obs_shape, config['n_actions'], config)

        # Initiate Logger ----------------
        self.traj_visualizer = TrajectoryVisualizer(self.env)
        self.logger = RLLogger(rows=3, cols=1, num_iterations=self.ITERATIONS)
        self.logger.add_lineplot('test_reward', xlabel='', ylabel='$R_{test}$', filter_window=30, display_raw=True, loc=(0, 1))
        self.logger.add_lineplot('train_reward', xlabel='', ylabel='$R_{train}$', filter_window=30, display_raw=True, loc=(1, 1))
        self.logger.add_lineplot('loss', xlabel='iter', ylabel='$Loss$', filter_window=30, display_raw=True, loc=(2, 1))
        self.logger.add_table('Params', config)
        self.logger.add_status()
        self.logger.add_button('Preview Game', callback=self.traj_visualizer.preview_qued_trajectory)

        # Initialize Variables ----------------
        self._epsilon = None
        self._rationality = None
        self._rshape_scale = None

        self.print_config(config)

    def print_config(self,config):
        for key, val in config.items():
            print(f'{key}={val}')

    def run(self):
        train_rewards = []
        train_losses = []
        # Main training Loop
        for it in range(self.ITERATIONS):
            self.logger.spin()

            # Training Step ##########################################
            # Set Iteration parameters
            self._epsilon = self.epsilon_sched[it]
            self._rationality = self.rationality_sched[it]
            self._rshape_scale = self.rshape_sched[it]

            # Perform Rollout
            self.logger.start_iteration()
            cum_reward, cum_shaped_rewards,mean_loss = self.training_rollout(it)
            if it>1: self.model.scheduler.step() # updates learning rate scheduler
            self.model.update_target()  # performs soft update of target network
            self.logger.end_iteration()

            print(f"Iteration {it} "
                  f"| train reward: {round(cum_reward, 3)} "
                  f"| shaped reward: {np.round(cum_shaped_rewards, 3)} "
                  f"| |memory| {self.model.memory_len} "
                  f"| |rshape| {round(self._rshape_scale, 3)} "
                  f"| rationality: {round(self._rationality, 3)}"
                  f"| epsilon: {round(self._epsilon, 3)} "
                  f"| LR={round(self.model.optimizer.param_groups[0]['lr'], 4)}"
                  )

            train_rewards.append(cum_reward + cum_shaped_rewards)
            train_losses.append(mean_loss)

            # Testing Step ##########################################
            # time4test = (it % self.test_interval == 0 and it > 2)
            time4test = (it % self.test_interval == 0)
            if time4test:
                test_rewards = []
                test_shaped_rewards = []
                self._rationality = self.test_rationality
                self._epsilon = 0
                self._rshape_scale = 1
                for test in range(self.N_tests):
                    test_reward, test_shaped_reward, state_history, action_history, aprob_history = self.test_rollout()
                    test_rewards.append(test_reward)
                    test_shaped_rewards.append(test_shaped_reward)
                    self.traj_visualizer.que_trajectory(state_history)

                self.logger.log(test_reward=[it, np.mean(test_rewards)],
                           train_reward=[it, np.mean(train_rewards)],
                           loss=[it, np.mean(train_losses)])
                self.logger.draw()
                print(f"\nTest: | nTests= {self.N_tests} "
                      f"| Ave Reward = {np.mean(test_rewards)} "
                      f"| Ave Shaped Reward = {np.mean(test_shaped_rewards)}"
                      # f"\n{action_history}\n"#, f"{aprob_history[0]}\n"
                      )
                train_rewards = []
                train_losses = []

        self.logger.wait_for_close(enable=True)

    ################################################################
    # Train/Test Rollouts   ########################################
    ################################################################
    def training_rollout(self,it):
        self.model.rationality = self._rationality
        self.env.reset()

        # Random start state if specified
        if it / self.ITERATIONS < self.perc_random_start:
            self.env.state = self.random_start_state()

        losses = []
        cum_reward = 0
        cum_shaped_reward = np.zeros(2)
        for t in count():
            # TODO: Verify if observing correctly
            obs = torch.tensor(self.mdp.get_lossless_encoding_vector(self.env.state), dtype=torch.float32, device=device).unsqueeze(0)
            joint_action, joint_action_idx, action_probs = self.model.choose_joint_action(obs, epsilon=self._epsilon)
            next_state_prospects = self.mdp.one_step_lookahead(self.env.state.deepcopy(),
                                                               joint_action=Action.ALL_JOINT_ACTIONS[joint_action_idx],
                                                               as_tensor=True, device=device)
            next_state, reward, done, info = self.env.step(joint_action)

            # Track reward traces
            shaped_rewards = self._rshape_scale * np.array(info["shaped_r_by_agent"])
            if self.shared_rew: shaped_rewards = np.mean(shaped_rewards)*np.ones(2)
            total_rewards =  np.array([reward + shaped_rewards]).flatten()
            cum_reward += reward
            cum_shaped_reward += shaped_rewards

            # Store in memory ----------------
            self.model.memory_double_push(state=obs,
                                        action=joint_action_idx,
                                        rewards = total_rewards,
                                        next_prospects=next_state_prospects,
                                        done = done)
            # Update model ----------------
            loss = self.model.update()
            if loss is not None: losses.append(loss)
            if done:  break
            self.env.state = next_state
        mean_loss = np.mean(losses)
        return cum_reward, cum_shaped_reward, mean_loss
    def test_rollout(self):
        self.model.rationality = self._rationality
        self.env.reset()

        test_reward = 0
        test_shaped_reward = 0
        state_history = [self.env.state.deepcopy()]
        action_history = []
        aprob_history = []

        for t in count():
            obs = torch.tensor(self.mdp.get_lossless_encoding_vector(self.env.state), dtype=torch.float32,
                               device=device).unsqueeze(0)
            joint_action, joint_action_idx, action_probs = self.model.choose_joint_action(obs, epsilon=self._epsilon)

            # obs = torch.tensor(self.mdp.get_lossless_encoding_vector(self.env.state), dtype=torch.float32, device=device).unsqueeze(0)
            # joint_action, joint_action_idx, action_probs = self.model.choose_joint_action(obs, epsilon=self._epsilon)
            next_state, reward, done, info = self.env.step(joint_action)

            # Track reward traces
            test_reward += reward
            test_shaped_reward += np.mean(info["shaped_r_by_agent"])*np.ones(2)

            # Track state-action history
            action_history.append(joint_action_idx)
            aprob_history.append(action_probs)
            state_history.append(next_state.deepcopy())

            if done:  break
            self.env.state = next_state
        return test_reward, test_shaped_reward, state_history, action_history, aprob_history

    ################################################################
    # State Randomizer #############################################
    ################################################################
    def random_start_state(self):
        state = self.add_random_start_loc()
        state = self.add_random_start_pot_state(state)
        state = self.add_random_held_obj(state)
        state = self.add_random_counter_state(state)
        return state
    def add_random_start_loc(self):
        random_state = self.mdp.get_random_start_state_fn(random_start_pos=True, rnd_obj_prob_thresh=0.0)()
        return random_state
    def add_random_start_pot_state(self,state,rnd_obj_prob_thresh=0.5):
        pots = self.mdp.get_pot_states(state)["empty"]
        for pot_loc in pots:
            p = np.random.rand()
            if p < rnd_obj_prob_thresh:
                n = int(np.random.randint(low=1, high=3))
                cooking_tick = np.random.randint(0, 20) if n == 3 else -1
                # cooking_tick = 0 if n == 3 else -1
                state.objects[pot_loc] = SoupState.get_soup(
                    pot_loc,
                    num_onions=n,
                    num_tomatoes=0,
                    cooking_tick=cooking_tick,
                )
        return state
    def add_random_held_obj(self,state,rnd_obj_prob_thresh=0.5):
        # For each player, add a random object with prob rnd_obj_prob_thresh
        for player in state.players:
            p = np.random.rand()
            if p < rnd_obj_prob_thresh:
                # Different objects have different probabilities
                obj = np.random.choice(["onion", "dish", "soup"], p=[0.6, 0.2, 0.2])
                if obj == "soup":
                    player.set_object( SoupState.get_soup(player.position, num_onions=3,  num_tomatoes=0, finished=True))
                else:
                    player.set_object(ObjectState(obj, player.position))
        return state

    def add_random_counter_state(self, state, rnd_obj_prob_thresh=0.05):
        counters = self.mdp.reachable_counters
        for counter_loc in counters:
            p = np.random.rand()
            if p < rnd_obj_prob_thresh:
                obj = np.random.choice(["onion", "dish", "soup"], p=[0.6, 0.2, 0.2])
                if obj == "soup":
                    state.add_object(SoupState.get_soup(counter_loc, num_onions=3, num_tomatoes=0, finished=True))
                else:
                    state.add_object(ObjectState(obj, counter_loc))
        return state

def main():

    config = {
        'ALGORITHM': 'Boltzmann_QRE-DDQN-OSA',
        'Date': datetime.now().strftime("%m/%d/%Y, %H:%M"),

        # Env Params ----------------
        # 'LAYOUT': "risky_coordination_ring", 'HORIZON': 200, 'ITERATIONS': 15_000,
        'LAYOUT': "coordination_ring_CLDE", 'HORIZON': 200, 'ITERATIONS': 15_000,

        # 'LAYOUT': "risky_cramped_room_CLCE", 'HORIZON': 200, 'ITERATIONS': 20_000,
        # 'LAYOUT': "cramped_room_CLCE", 'HORIZON': 200, 'ITERATIONS': 20_000,
        # 'LAYOUT': "super_cramped_room", 'HORIZON': 200, 'ITERATIONS': 10_000,
        # 'LAYOUT': "risky_super_cramped_room", 'HORIZON': 200, 'ITERATIONS': 10_000,
        'AGENT': None,                  # name of agent object (computed dynamically)
        "obs_shape": None,                  # computed dynamically based on layout
        "perc_random_start": 0.9,          # percentage of ITERATIONS with random start states

        # Learning Params ----------------
        'epsilon_sched': [0.1,0.1,5000],         # epsilon-greedy range (start,end)
        'rshape_sched': [1,0,5_000],     # rationality level range (start,end)
        'rationality_sched': [0.0,5,5000],
        'lr_sched': [1e-2,1e-4,5_000],
        'test_rationality': 5,          # rationality level for testing
        'gamma': 0.95,                      # discount factor
        'tau': 0.005,                       # soft update weight of target network
        "num_hidden_layers": 3,             # MLP params
        "size_hidden_layers": 256,#32,      # MLP params
        "device": device,
        "minibatch_size":256,          # size of mini-batches
        "replay_memory_size": 20_000,   # size of replay memory
        'clip_grad': 100,

    }
    # config['LAYOUT'] = "cramped_room_CLCE"

    # BEST ##########################
    # config['replay_memory_size'] = 30_000
    # config['epsilon_sched'] = [1.0, 0.1, 10_000]
    # config['rshape_sched'] = [1, 0, 10_000]
    # config['rationality_sched'] = [5.0, 5.0, 5_000]
    # config['perc_random_start'] = 0.9
    # config['test_rationality'] = config['rationality_sched'][1]
    # config['lr'] = 1e-4
    # config['tau'] = 0.01
    # config['lr_warmup_iter'] = 5000
    # config['lr_warmup_scale'] = 100
    ###############################

    # Top Left
    # config['replay_memory_size'] = 30_000
    # config['epsilon_sched'] = [1.0, 0.1, 10_000]
    # config['rshape_sched'] = [1, 0, 10_000]
    # config['rationality_sched'] = [2.0, 2.0, 10_000]
    # config['perc_random_start'] = 0.9
    # config['test_rationality'] = config['rationality_sched'][1]
    # config['lr'] = 1e-4
    # config['tau'] = 0.01
    # config['lr_warmup_iter'] = 5000
    # config['lr_warmup_scale'] = 100
    # config['num_hidden_layers'] = 5
    # config['size_hidden_layers'] = 128


    # Bottom Left
    # config['ITERATIONS'] = 10_000
    # config['replay_memory_size'] = 30_000
    # config['epsilon_sched'] = [1.0, 0.1, 10_000]
    # config['rshape_sched'] = [1, 0, 10_000]
    # config['rationality_sched'] = [5.0, 5.0, 5_000]
    # config['perc_random_start'] = 0.9
    # config['test_rationality'] = config['rationality_sched'][1]
    # config['tau'] = 0.01
    # config['lr_sched'] = [1e-2,1e-5,2_000]
    # config['num_hidden_layers'] = 5
    # config['size_hidden_layers'] = 128

    # # Top Right
    # config['LAYOUT'] = "risky_coordination_ring"
    config['ITERATIONS'] = 10_000
    config['replay_memory_size'] = 30_000
    config['epsilon_sched'] = [1.0, 0.1, 10_000]
    config['rshape_sched'] = [1, 0, 10_000]
    config['rationality_sched'] = [5.0, 5.0, 5_000]
    config['perc_random_start'] = 0.9
    config['test_rationality'] = config['rationality_sched'][1]
    config['tau'] = 0.001
    config['lr_sched'] = [1e-2, 1e-5, 2_000]
    config['num_hidden_layers'] = 5
    config['size_hidden_layers'] = 128


    # bottom Right
    # config['LAYOUT'] = "risky_coordination_ring"
    # config['replay_memory_size'] = 30_000
    # config['epsilon_sched'] = [1.0, 0.1, 10_000]
    # config['rshape_sched'] = [1, 0, 10_000]
    # config['rationality_sched'] = [5.0, 5.0, 5_000]
    # config['perc_random_start'] = 0.9
    # config['test_rationality'] = config['rationality_sched'][1]
    # config['lr'] = 1e-5
    # config['tau'] = 0.01
    # config['lr_warmup_iter'] = 5000
    # config['lr_warmup_scale'] = 100
    # config['num_hidden_layers'] = 6
    # config['size_hidden_layers'] = 128

    Trainer(SelfPlay_QRE_OSA,config).run()

    # config['cpt_params']= {'b': 0.0, 'lam': 1.0,
    #                'eta_p': 1., 'eta_n': 1.,
    #                'delta_p': 1., 'delta_n': 1.}
    # # config['cpt_params']= {'b': 0.4, 'lam': 1.0,
    # #                'eta_p': 0.88, 'eta_n': 0.88,
    # #                'delta_p': 0.61, 'delta_n': 0.69}
    # Trainer(SelfPlay_QRE_OSA_CPT,config).run()




if __name__ == "__main__":
    main()