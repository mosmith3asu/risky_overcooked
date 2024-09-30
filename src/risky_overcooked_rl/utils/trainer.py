import numpy as np
from risky_overcooked_rl.utils.deep_models import device
from risky_overcooked_rl.utils.rl_logger import RLLogger,TrajectoryVisualizer, TrajectoryHeatmap
from risky_overcooked_rl.utils.model_manager import get_absolute_save_dir
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld,SoupState, ObjectState
from risky_overcooked_py.mdp.actions import Action
from itertools import count
from risky_overcooked_rl.utils.state_utils import FeasibleActionManager
import torch


import random
from datetime import datetime
debug = False
from collections import deque

class Trainer:
    def __init__(self,model_object,config):
        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)

        # Load default and parse custom config ---
        config['obs_shape'] = None # defined later
        config['device'] = device
        config['AGENT'] = model_object.__name__
        config['Date'] = datetime.now().strftime("%m_%d_%Y-%H_%M")
        self.config = config
        self.cpt_params = config['cpt_params']
        self.timestamp = config['Date']

        # Generate MDP and environment----------------
        self.LAYOUT = config['LAYOUT']
        self.HORIZON = config['HORIZON']
        self.device = config['device']
        self.shared_rew = False
        self.curriculum = None

        self.ITERATIONS = config['ITERATIONS']
        self.mdp = OvercookedGridworld.from_layout_name(self.LAYOUT)
        self.mdp.p_slip = config['p_slip']
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=self.HORIZON)
        self.N_tests = 1
        self.test_interval = 10  # test every n iterations
        self.feasible_action = FeasibleActionManager(self.env)

        # Define Parameter Schedules ------------------
        self.init_sched(config)

        # Initialize policy and target networks ----------------
        obs_shape = self.mdp.get_lossless_encoding_vector_shape()
        config['obs_shape'] = obs_shape
        n_actions = 36

        if config['loads'] == 'rational':
            rational_fname = f"{self.LAYOUT}_pslip{str(self.mdp.p_slip).replace('.', '')}__rational__"
            self.model = model_object.from_file(obs_shape, n_actions, config,rational_fname)
        elif config['loads'] == '': self.model = model_object(obs_shape, n_actions, config)
        else: raise ValueError(f"Invalid load option: {config['loads']}")

        # Initiate Logger and Managers ----------------
        self.traj_visualizer = TrajectoryVisualizer(self.env)
        self.traj_heatmap = TrajectoryHeatmap(self.env)
        self.logger = RLLogger(rows=3, cols=1, num_iterations=self.ITERATIONS)
        self.logger.add_lineplot('test_reward', xlabel='', ylabel='$R_{test}$', filter_window=30, display_raw=True, loc=(0, 1))
        self.logger.add_lineplot('train_reward', xlabel='', ylabel='$R_{train}$', filter_window=30, display_raw=True, loc=(1, 1))
        self.logger.add_lineplot('loss', xlabel='iter', ylabel='$Loss$', filter_window=30, display_raw=True, loc=(2, 1))
        self.logger.add_checkpoint_line()
        self.logger.add_table('Params', config)
        self.logger.add_status()
        self.logger.add_button('Preview', callback=self.traj_visualizer.preview_qued_trajectory)
        self.logger.add_button('Heatmap', callback=self.traj_heatmap.preview)
        self.logger.add_button('Save ', callback=self.save)


        # Initialize Variables ----------------
        self._epsilon = None
        self._rationality = None
        self._rshape_scale = None

        # Checkpointing/Saving utils ----------------
        self.checkpoint_score = 0
        self.min_checkpoint_score = 20
        self.checkpoint_mem = 10
        self.has_checkpointed = False
        self.train_rewards = deque(maxlen=self.checkpoint_mem)
        self.test_rewards = deque(maxlen=self.checkpoint_mem)

        # Report ----------------
        self.print_config(config)

    @property
    def fname(self):
        if (self.cpt_params['b'] == 0
            and self.cpt_params['lam'] == 1.0
            and self.cpt_params['eta_p'] == 1.0
            and self.cpt_params['eta_n'] == 1.0
            and self.cpt_params['delta_p'] == 1.0
            and self.cpt_params['delta_n'] == 1.0):
            h = f"{self.LAYOUT}" \
                f"_pslip{str(self.mdp.p_slip).replace('.', '')}" \
                f"__rational" \
                f"__{self.config['Date']}"
        else:
            h = f"{self.LAYOUT}" \
                f"_pslip{str(self.mdp.p_slip).replace('.', '')}" \
                f"__b{str(self.cpt_params['b']).replace('.', '')}" \
                f"_lam{str(self.cpt_params['lam']).replace('.', '')}" \
                f"_etap{str(self.cpt_params['eta_p']).replace('.', '')}" \
                f"_etan{str(self.cpt_params['eta_n']).replace('.', '')}" \
                f"_deltap{str(self.cpt_params['delta_p']).replace('.', '')}" \
                f"_deltan{str(self.cpt_params['delta_n']).replace('.', '')}" \
                f"__{self.config['Date']}"
        return h

    def print_config(self,config):
        for key, val in config.items():
            print(f'{key}={val}')
    def init_sched(self,config,eps_decay = 1,rshape_decay=1):
        # def exponential_decay(N0, Nf, t, T):
        #     w = 0.75
        #     if t > T: return Nf
        #     return N0 * (Nf / N0) ** ((t / T) ** w)
        def exponential_decay(N0, Nf, t, T, cycle=True):
            w = 0.75
            if t > T:
                if cycle: # cycle through min and max decay after final iteration reached
                    _t = t % T if int(t / T) % 2 == 0 else T - t % T
                    return (N0 * (Nf / N0) ** ((_t / T) ** w))
                else: return Nf
            return N0 * (Nf / N0) ** ((t / T) ** w)

        EPS_START, EPS_END, EPS_DUR = config['epsilon_sched']
        RAT_START, RAT_END, RAT_DUR = config['rationality_sched']
        RSHAPE_START, RSHAPE_END, RSHAPE_DUR = config['rshape_sched']
        RSTART_START, RSTART_END, RSTART_DUR = config['rand_start_sched']

        EPS_START = EPS_START * eps_decay
        RSHAPE_START = RSHAPE_START * rshape_decay

        self.test_rationality = RAT_END  # config['test_rationality']
        self.rationality_sched = np.hstack(
            [np.linspace(RAT_START, RAT_END, RAT_DUR), RAT_END * np.ones(self.ITERATIONS - RAT_DUR)])
        # self.epsilon_sched = np.hstack(
        #     [np.linspace(EPS_START, EPS_END, EPS_DUR), EPS_END * np.ones(self.ITERATIONS - EPS_DUR)])
        self.epsilon_sched = [exponential_decay(N0=EPS_START, Nf=EPS_END, t=t, T=EPS_DUR) for t in range(self.ITERATIONS)]

        self.rshape_sched = np.hstack(
            [np.linspace(RSHAPE_START, RSHAPE_END, RSHAPE_DUR), RSHAPE_END * np.ones(self.ITERATIONS - RSHAPE_DUR)])
        self.random_start_sched = np.hstack(
            [np.linspace(RSTART_START, RSTART_END, RSTART_DUR), RSTART_END * np.ones(self.ITERATIONS - RSTART_DUR)])

    def run(self):
        train_rewards = []
        train_losses = []
        # Main training Loop
        for it in range(self.ITERATIONS):
            self.logger.spin()

            # Training Step ##########################################
            # Set Iteration parameters

            self._p_rand_start = self.random_start_sched[it]

            # Perform Rollout
            self.logger.start_iteration()

            cum_reward, cum_shaped_rewards,rollout_info =\
                self.training_rollout(it,rationality=self.rationality_sched[it],
                                      epsilon = self.epsilon_sched[it],
                                      rshape_scale= self.rshape_sched[it],
                                      p_rand_start=self.random_start_sched[it])

            if it>1: self.model.scheduler.step() # updates learning rate scheduler
            self.model.update_target()  # performs soft update of target network
            self.logger.end_iteration()

            # slips = rollout_info['onion_slips'] + rollout_info['dish_slips'] + rollout_info['soup_slips']
            risks = rollout_info['onion_risked'] + rollout_info['dish_risked'] + rollout_info['soup_risked']
            handoffs = rollout_info['onion_handoff'] + rollout_info['dish_handoff'] + rollout_info['soup_handoff']

            print(f"Iteration {it} "
                  f"| train reward:{round(cum_reward, 3)} "
                  f"| shaped reward:{np.round(cum_shaped_rewards, 3)} "
                  f"| loss:{round(rollout_info['mean_loss'], 3)} "
                  # f"| slips:{slips} "
                  f"| risks:{risks} "
                  f"| handoffs:{handoffs} "
                  f" |"
                  f"| mem:{self.model.memory_len} "
                  f"| rshape:{round(self.rshape_sched[it], 3)} "
                  f"| rat:{round(self.rationality_sched[it], 3)}"
                  f"| eps:{round(self.epsilon_sched[it], 3)} "
                  f"| LR={round(self.model.optimizer.param_groups[0]['lr'], 4)}"
                  f"| rstart={round(self.random_start_sched[it], 3)}"
                  )

            train_rewards.append(cum_reward + cum_shaped_rewards)
            train_losses.append(rollout_info['mean_loss'])

            # Testing Step ##########################################
            # time4test = (it % self.test_interval == 0 and it > 2)
            time4test = (it % self.test_interval == 0)
            if time4test:

                # Rollout test episodes ----------------------
                test_rewards = []
                test_shaped_rewards = []
                for test in range(self.N_tests):
                    test_reward, test_shaped_reward, state_history, action_history, aprob_history =\
                        self.test_rollout(rationality=self.test_rationality)
                    test_rewards.append(test_reward)
                    test_shaped_rewards.append(test_shaped_reward)
                    if not self.has_checkpointed:
                        self.traj_visualizer.que_trajectory(state_history)
                        self.traj_heatmap.que_trajectory(state_history)

                # Checkpointing ----------------------
                self.test_rewards.append(np.mean(test_rewards))  # for checkpointing
                self.train_rewards.append(np.mean(train_rewards))  # for checkpointing
                if self.checkpoint(it):  # check if should checkpoint
                    self.traj_visualizer.que_trajectory(state_history) # load preview of checkpointed trajectory
                    self.traj_heatmap.que_trajectory(state_history)
                # Logging ----------------------
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
    def training_rollout(self,it,rationality,epsilon,rshape_scale,p_rand_start=0):

        self.model.rationality = rationality
        self.env.reset()

        # Random start state if specified
        # if it / self.ITERATIONS < self.perc_random_start:
        if np.random.sample() < p_rand_start:
            self.env.state = self.random_start_state()

        losses = []
        cum_reward = 0
        cum_shaped_reward = np.zeros(2)

        rollout_info = {
            'onion_risked': np.zeros([1, 2]),
            'onion_pickup': np.zeros([1, 2]),
            'onion_drop': np.zeros([1, 2]),
            'dish_risked': np.zeros([1, 2]),
            'dish_pickup': np.zeros([1, 2]),
            'dish_drop': np.zeros([1, 2]),
            'soup_pickup': np.zeros([1, 2]),
            'soup_delivery': np.zeros([1, 2]),

            'soup_risked': np.zeros([1, 2]),
            'onion_slip': np.zeros([1, 2]),
            'dish_slip': np.zeros([1, 2]),
            'soup_slip': np.zeros([1, 2]),
            'onion_handoff': np.zeros([1, 2]),
            'dish_handoff': np.zeros([1, 2]),
            'soup_handoff': np.zeros([1, 2]),
            'mean_loss': 0
        }

        for t in count():
            obs = self.mdp.get_lossless_encoding_vector_astensor(self.env.state,device=device).unsqueeze(0)
            feasible_JAs = self.feasible_action.get_feasible_joint_actions(self.env.state,as_joint_idx=True)
            joint_action, joint_action_idx, action_probs = self.model.choose_joint_action(obs,
                                                                                          epsilon=epsilon,
                                                                                          feasible_JAs = feasible_JAs)
            next_state_prospects = self.mdp.one_step_lookahead(self.env.state.deepcopy(),
                                                               joint_action=Action.ALL_JOINT_ACTIONS[joint_action_idx],
                                                               as_tensor=True, device=device)
            next_state, reward, done, info = self.env.step(joint_action,get_mdp_info=True)

            for key in rollout_info.keys():
                if key not in ['mean_loss']:
                    rollout_info[key] += np.array(info['mdp_info']['event_infos'][key])

            # Track reward traces
            shaped_rewards = rshape_scale * np.array(info["shaped_r_by_agent"])
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
    def test_rollout(self,rationality,epsilon=0,rshape_scale=1):
        self.model.model.eval()
        self.model.target.eval()
        self.model.rationality = rationality
        self.env.reset()

        test_reward = 0
        test_shaped_reward = 0
        state_history = [self.env.state.deepcopy()]
        action_history = []
        aprob_history = []

        for t in count():
            obs = self.mdp.get_lossless_encoding_vector_astensor(self.env.state,device=device).unsqueeze(0)
            joint_action, joint_action_idx, action_probs = self.model.choose_joint_action(obs, epsilon=epsilon)
            next_state, reward, done, info = self.env.step(joint_action)

            # Track reward traces
            test_reward += reward
            test_shaped_reward += rshape_scale*np.mean(info["shaped_r_by_agent"])*np.ones(2)

            # Track state-action history
            action_history.append(joint_action_idx)
            aprob_history.append(action_probs)
            state_history.append(next_state.deepcopy())

            if done:  break
            self.env.state = next_state
        self.model.model.train()
        self.model.target.train()
        return test_reward, test_shaped_reward, state_history, action_history, aprob_history

    ################################################################
    # State Randomizer #############################################
    ################################################################
    def random_start_state(self):
        state = self.add_random_start_loc()
        state = self.add_random_start_pot_state(state)
        state = self.add_random_held_obj(state)
        # state = self.add_random_counter_state(state)
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
        if obj == "soup": player.set_object(SoupState.get_soup(player.position, num_onions=3, num_tomatoes=0, finished=True))
        else: player.set_object(ObjectState(obj, player.position))
        return player

    ################################################################
    # Save Utils       #############################################
    ################################################################
    def checkpoint(self,it):
        if len(self.train_rewards) == self.checkpoint_mem:
            ave_train = np.mean(self.train_rewards)
            ave_test = np.mean(self.test_rewards)
            # score = (ave_train + ave_test)/2
            score = ave_test
            if score > self.min_checkpoint_score and score > self.checkpoint_score:
                print(f'\nCheckpointing model at iteration {it} with score {score}...\n')
                self.model.update_checkpoint()
                self.logger.update_checkpiont_line(it)
                self.checkpoint_score = score
                self.has_checkpointed = True
                return True
        return False


    def package_model_info(self,rational=False):
        model_info = {
            'timestamp': self.timestamp,
            'layout': self.LAYOUT,
            'p_slip': self.mdp.p_slip,
            'b': self.cpt_params['b'] if not rational else 0.0,
            'lam': self.cpt_params['lam'] if not rational else 1.0,
            'eta_p': self.cpt_params['eta_p'] if not rational else 1.0,
            'eta_n': self.cpt_params['eta_n'] if not rational else 1.0,
            'delta_p': self.cpt_params['delta_p'] if not rational else 1.0,
            'delta_n': self.cpt_params['delta_n'] if not rational else 1.0,
        }
        return model_info
    def save(self,*args):
        # find saved models absolute dir -------------
        print(f'\n\nSaving model to {self.fname}...')
        dir = get_absolute_save_dir()
        torch.save(self.model.checkpoint_model.state_dict(), dir + f"{self.fname}.pt")
        # model_info = self.package_model_info()
        # self.model_manager.save(model,model_info,self.fname)
        self.logger.save_fig(f"./models/{self.fname}.png")
        print(f'finished\n\n')


if __name__ == "__main__":
    raise NotImplementedError