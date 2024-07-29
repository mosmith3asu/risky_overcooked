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
import random
from datetime import datetime
debug = False

class Trainer:
    def __init__(self,model_object,config):
        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)

        # Generate MDP and environment----------------
        config['AGENT'] = model_object.__name__
        LAYOUT = config['LAYOUT']
        HORIZON = config['HORIZON']
        self.device = config['device']
        self.shared_rew = False

        self.ITERATIONS = config['ITERATIONS']
        self.mdp = OvercookedGridworld.from_layout_name(LAYOUT)
        self.mdp.p_slip = config['p_slip']
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=HORIZON)
        # self.perc_random_start = config['perc_random_start']
        self.N_tests = 1
        self.test_interval = 10  # test every n iterations

        # Define Parameter Schedules ------------------
        EPS_START, EPS_END, EPS_DUR = config['epsilon_sched']
        RAT_START, RAT_END, RAT_DUR = config['rationality_sched']
        RSHAPE_START, RSHAPE_END, RSHAPE_DUR = config['rshape_sched']
        RSTART_START, RSTART_END, RSTART_DUR = config['rand_start_sched']
        self.test_rationality = RAT_END #config['test_rationality']
        self.rationality_sched = np.hstack( [np.linspace(RAT_START, RAT_END, RAT_DUR), RAT_END * np.ones(self.ITERATIONS - RAT_DUR)])
        self.epsilon_sched = np.hstack([np.linspace(EPS_START, EPS_END, EPS_DUR), EPS_END * np.ones(self.ITERATIONS - EPS_DUR)])
        self.rshape_sched = np.hstack([np.linspace(RSHAPE_START, RSHAPE_END, RSHAPE_DUR), RSHAPE_END * np.ones(self.ITERATIONS - RSHAPE_DUR)])
        self.random_start_sched = np.hstack([np.linspace(RSTART_START, RSTART_END, RSTART_DUR), RSTART_END * np.ones(self.ITERATIONS - RSTART_DUR)])

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
            self._p_rand_start = self.random_start_sched[it]

            # Perform Rollout
            self.logger.start_iteration()
            cum_reward, cum_shaped_rewards,rollout_info = self.training_rollout(it)
            if it>1: self.model.scheduler.step() # updates learning rate scheduler
            self.model.update_target()  # performs soft update of target network
            self.logger.end_iteration()

            slips = rollout_info['onion_slips'] + rollout_info['dish_slips'] + rollout_info['soup_slips']

            print(f"Iteration {it} "
                  f"| train reward:{round(cum_reward, 3)} "
                  f"| shaped reward:{np.round(cum_shaped_rewards, 3)} "
                  f"| loss:{round(rollout_info['mean_loss'], 3)} "
                  f"| slips:{slips} "
                  f" |"
                  f"| mem:{self.model.memory_len} "
                  f"| rshape:{round(self._rshape_scale, 3)} "
                  f"| rat:{round(self._rationality, 3)}"
                  f"| eps:{round(self._epsilon, 3)} "
                  f"| LR={round(self.model.optimizer.param_groups[0]['lr'], 4)}"
                  f"| rstart={round(self._p_rand_start, 3)}"
                  )

            train_rewards.append(cum_reward + cum_shaped_rewards)
            train_losses.append(rollout_info['mean_loss'])

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
        # if it / self.ITERATIONS < self.perc_random_start:
        if np.random.sample() < self._p_rand_start:
            self.env.state = self.random_start_state()

        losses = []
        cum_reward = 0
        cum_shaped_reward = np.zeros(2)

        rollout_info = {
            'onion_slips': np.zeros(2),
            'dish_slips': np.zeros(2),
            'soup_slips': np.zeros(2),
            'mean_loss': 0
        }

        for t in count():
            obs = self.mdp.get_lossless_encoding_vector_astensor(self.env.state,device=device).unsqueeze(0)

            joint_action, joint_action_idx, action_probs = self.model.choose_joint_action(obs, epsilon=self._epsilon)
            next_state_prospects = self.mdp.one_step_lookahead(self.env.state.deepcopy(),
                                                               joint_action=Action.ALL_JOINT_ACTIONS[joint_action_idx],
                                                               as_tensor=True, device=device)
            next_state, reward, done, info = self.env.step(joint_action,get_mdp_info=True)
            rollout_info['onion_slips'] += np.array(info['mdp_info']['event_infos']['onion_slip'])
            rollout_info['dish_slips'] += np.array(info['mdp_info']['event_infos']['dish_slip'])
            rollout_info['soup_slips'] += np.array(info['mdp_info']['event_infos']['soup_slip'])


            # evemts = info['event_infos']
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
        rollout_info['mean_loss'] = np.mean(losses)
        return cum_reward, cum_shaped_reward, rollout_info
    def test_rollout(self):
        self.model.rationality = self._rationality
        self.env.reset()

        test_reward = 0
        test_shaped_reward = 0
        state_history = [self.env.state.deepcopy()]
        action_history = []
        aprob_history = []

        for t in count():
            # obs = torch.tensor(self.mdp.get_lossless_encoding_vector(self.env.state), dtype=torch.float32,
            #                    device=device).unsqueeze(0)
            obs = self.mdp.get_lossless_encoding_vector_astensor(self.env.state,device=device).unsqueeze(0)

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
                self.add_held_obj(player, obj)
        return state
    def add_random_counter_state(self, state, rnd_obj_prob_thresh=0.025):
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

    def add_held_obj(self,player,obj):
        if obj == "soup":
            player.set_object(SoupState.get_soup(player.position, num_onions=3, num_tomatoes=0, finished=True))
        else:
            player.set_object(ObjectState(obj, player.position))
        return player

def main():

    config = {
        'ALGORITHM': 'Boltzmann_QRE-DDQN-OSA',
        'Date': datetime.now().strftime("%m/%d/%Y, %H:%M"),

        # Env Params ----------------
        'LAYOUT': "coordination_ring_CLDE", 'HORIZON': 200, 'ITERATIONS': 15_000,
        'AGENT': None,                  # name of agent object (computed dynamically)
        "obs_shape": None,                  # computed dynamically based on layout
        # "shared_rew": False,                # shared reward for both agents
        "p_slip": 0.25,
        # Learning Params ----------------
        "rand_start_sched": [1.0, 0.5, 10_000],  # percentage of ITERATIONS with random start states
        'epsilon_sched': [0.1,0.1,5000],         # epsilon-greedy range (start,end)
        'rshape_sched': [1,0,5_000],     # rationality level range (start,end)
        'rationality_sched': [0.0,5,5000],
        'lr_sched': [1e-2,1e-4,3_000],
        # 'test_rationality': 5,          # rationality level for testing
        'gamma': 0.95,                      # discount factor
        'tau': 0.005,                       # soft update weight of target network
        "num_hidden_layers": 3,             # MLP params
        "size_hidden_layers": 256,#32,      # MLP params
        "device": device,
        "minibatch_size":256,          # size of mini-batches
        "replay_memory_size": 30_000,   # size of replay memory
        'clip_grad': 100,

    }
    # config['LAYOUT'] = "cramped_room_CLCE"

    # BEST ##########################
    # config['replay_memory_size'] = 30_000
    # config['epsilon_sched'] = [1.0, 0.15, 10_000]
    # config['rshape_sched'] = [1, 0, 10_000]
    # config['rationality_sched'] = [5.0, 5.0, 10_000]
    # config['lr_sched'] = [1e-2, 1e-4, 3_000]
    # config['perc_random_start'] = 0.9
    # config['tau'] = 0.01
    # config['num_hidden_layers'] = 5
    # config['size_hidden_layers'] = 256
    # config['shared_rew'] = False
    # config['gamma'] = 0.95
    ###############################

    # Top Left
    config['ITERATIONS'] = 30_000
    config['LAYOUT'] = "risky_coordination_ring"
    config['replay_memory_size'] = 30_000
    config['epsilon_sched'] = [1.0, 0.1, 15_000]
    config['rshape_sched'] = [1, 0, 10_000]
    config['rationality_sched'] = [5.0, 5.0, 10_000]
    config['lr_sched'] = [1e-2, 1e-4, 3_000]
    config["rand_start_sched"]= [1.0, 0.75, 10_000]  # percentage of ITERATIONS with random start states
    config['tau'] = 0.01
    config['num_hidden_layers'] = 5
    config['size_hidden_layers'] = 256
    config['gamma'] = 0.95
    config['p_slip'] = 0.25
    config['note'] = 'medium risk + random chance start'
    Trainer(SelfPlay_QRE_OSA, config).run()



    # # Bottom Left
    # config['LAYOUT'] = "risky_coordination_ring"
    # config['ITERATIONS'] = 30_000
    # config['replay_memory_size'] = 30_000
    # config['epsilon_sched'] = [1.0, 0.15, 10_000]
    # config['rshape_sched'] = [1, 0, 10_000]
    # config['rationality_sched'] = [5.0, 5.0, 10_000]
    # config['lr_sched'] = [1e-2, 1e-4, 3_000]
    # config["rand_start_sched"]= [1.0, 0.5, 15_000] #config['perc_random_start'] = 0.9
    # config['tau'] = 0.01
    # config['num_hidden_layers'] = 5
    # config['size_hidden_layers'] = 256
    # config['gamma'] = 0.97
    # config['p_slip'] = 0.1
    # config['note'] = 'minimal risk + chance start + increased gamma + lower LR'
    # Trainer(SelfPlay_QRE_OSA, config).run()
    # config['LAYOUT'] = "risky_coordination_ring"
    # config['replay_memory_size'] = 30_000
    # config['epsilon_sched'] = [1.0, 0.15, 10_000]
    # config['rshape_sched'] = [1, 0, 10_000]
    # config['rationality_sched'] = [5.0, 5.0, 10_000]
    # config['lr_sched'] = [1e-2, 1e-4, 3_000]
    # config["rand_start_sched"]= [1.0, 0.75, 10_000]  # percentage of ITERATIONS with random start states # config['perc_random_start'] = 0.9
    # config['tau'] = 0.01
    # config['num_hidden_layers'] = 5
    # config['size_hidden_layers'] = 256
    # config['gamma'] = 0.95
    # config['p_slip'] = 0.1
    # config['note'] = 'minimal risk + trivial cpt+ chance start'
    # config['cpt_params'] = {'b': 0.0, 'lam': 1.0,
    #                         'eta_p': 1., 'eta_n': 1.,
    #                         'delta_p': 1., 'delta_n': 1.}
    # # config['cpt_params']= {'b': 0.4, 'lam': 1.0,
    # #                'eta_p': 0.88, 'eta_n': 0.88,
    # #                'delta_p': 0.61, 'delta_n': 0.69}
    # Trainer(SelfPlay_QRE_OSA_CPT, config).run()

    # config['replay_memory_size'] = 30_000
    # config['epsilon_sched'] = [1.0, 0.2, 8_000]
    # config['rshape_sched'] = [1, 0, 10_000]
    # config['rationality_sched'] = [5.0, 5.0, 10_000]
    # config['lr_sched'] = [1e-2, 1e-4, 3_000]
    # config['perc_random_start'] = 0.9
    # config['tau'] = 0.01
    # config['num_hidden_layers'] = 5
    # config['size_hidden_layers'] = 256
    # config['shared_rew'] = False
    # config['gamma'] = 0.95
    # config['note'] = 'added collab reward shaping'
    # Trainer(SelfPlay_QRE_OSA, config).run()
    # # config['replay_memory_size'] = 30_000
    # # config['epsilon_sched'] = [1.0, 0.15, 10_000]
    # # config['rshape_sched'] = [1, 0, 10_000]
    # # config['rationality_sched'] = [5.0, 5.0, 10_000]
    # # config['lr_sched'] = [1e-2, 1e-4, 5_000]
    # # config['perc_random_start'] = 0.9
    # # config['test_rationality'] = config['rationality_sched'][1]
    # # config['tau'] = 0.01
    # # config['num_hidden_layers'] = 6
    # # config['size_hidden_layers'] = 128
    # # config['shared_rew'] = False
    # # config['gamma'] = 0.95
    # # config['note'] = 'increased depth'
    # # Trainer(SelfPlay_QRE_OSA, config).run()


    # # Top Right
    # config['LAYOUT'] = "risky_coordination_ring"
    # config['ITERATIONS'] = 30_000
    # config['replay_memory_size'] = 30_000
    # config['epsilon_sched'] = [1.0, 0.15, 10_000]
    # config['rshape_sched'] = [1, 0, 10_000]
    # config['rationality_sched'] = [5.0, 5.0, 10_000]
    # config['lr_sched'] = [1e-2, 1e-4, 4_000]
    # config["rand_start_sched"] = [0.0, 0.00, 15_000]  # config['perc_random_start'] = 0.9
    # config['tau'] = 0.01
    # config['num_hidden_layers'] = 5
    # config['size_hidden_layers'] = 256
    # config['gamma'] = 0.97
    # config['p_slip'] = 0.99
    # config['note'] = 'max risk + (test handoff)'
    # Trainer(SelfPlay_QRE_OSA, config).run()

    # bottom Right
    # config['LAYOUT'] = "risky_coordination_ring"
    # config['ITERATIONS'] = 30_000
    # config['replay_memory_size'] = 30_000
    # config['epsilon_sched'] = [1.0, 0.15, 10_000]
    # config['rshape_sched'] = [1, 0, 10_000]
    # config['rationality_sched'] = [5.0, 5.0, 10_000]
    # config['lr_sched'] = [1e-2, 1e-4, 3_000]
    # config["rand_start_sched"] = [1.5, 0.05, 15_000]  # config['perc_random_start'] = 0.9
    # config['tau'] = 0.01
    # config['num_hidden_layers'] = 5
    # config['size_hidden_layers'] = 256
    # config['gamma'] = 0.95
    # config['p_slip'] = 0.1
    # config['note'] = 'minimal risk + chance start'
    # Trainer(SelfPlay_QRE_OSA, config).run()
    # config['LAYOUT'] = "risky_coordination_ring"
    # config['ITERATIONS'] = 30_000
    # config['replay_memory_size'] = 30_000
    # config['epsilon_sched'] = [1.0, 0.15, 10_000]
    # config['rshape_sched'] = [1, 0, 10_000]
    # config['rationality_sched'] = [5.0, 5.0, 10_000]
    # config['lr_sched'] = [1e-2, 1e-4, 2_000]
    # config['perc_random_start'] = 0.9
    # config['tau'] = 0.01
    # config['num_hidden_layers'] = 5
    # config['size_hidden_layers'] = 256
    # config['gamma'] = 0.95
    # config['p_slip'] = 0.1
    # config['note'] = 'minimal risk + 90% chance start'
    # Trainer(SelfPlay_QRE_OSA, config).run()

    # config['cpt_params']= {'b': 0.0, 'lam': 1.0,
    #                'eta_p': 1., 'eta_n': 1.,
    #                'delta_p': 1., 'delta_n': 1.}
    # # config['cpt_params']= {'b': 0.4, 'lam': 1.0,
    # #                'eta_p': 0.88, 'eta_n': 0.88,
    # #                'delta_p': 0.61, 'delta_n': 0.69}
    # Trainer(SelfPlay_QRE_OSA_CPT,config).run()




if __name__ == "__main__":
    main()