import numpy as np
from risky_overcooked_rl.utils.rl_logger_v2 import RLLogger_V2
from risky_overcooked_rl.utils.rl_logger import RLLogger,TrajectoryVisualizer, TrajectoryHeatmap
from risky_overcooked_rl.algorithms.DDQN import get_absolute_save_dir
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
    def __init__(self,model_object,master_config):

        # Update configs with runtime values --------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        master_config['agents']['model']['device'] = self.device
        master_config['agents']['type'] = model_object.__name__
        master_config['trainer']['obs_shape'] = None # defined later
        self.timestamp = datetime.now().strftime("%m_%d_%Y-%H_%M")
        master_config['save']['date'] = self.timestamp
        self.master_config = master_config

        # Parse Sub Configurations -----------------------
        env_config = master_config['env']
        trainer_config = master_config['trainer']
        agents_config = master_config['agents']
        logger_config = master_config['logger']
        save_config = master_config['save']


        # Set random seeds -----------------
        seed = trainer_config['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        # Parse Trainer Configuration -------------------

        self.ITERATIONS = trainer_config['ITERATIONS']
        self.warmup_transitions = trainer_config['warmup_transitions']
        self.N_tests = trainer_config['N_tests']
        self.test_interval =  trainer_config['test_interval']
        enable_feasible_actions = trainer_config['feasible_actions']
        n_actions = trainer_config['joint_action_shape']

        # Instantiate MDP and Environment ----------------
        layout = env_config['LAYOUT']
        time_cost = env_config['time_cost']
        p_slip = env_config['p_slip']
        horizon = env_config['HORIZON']
        neglect_boarders = env_config['neglect_boarders']
        overwrite = {}
        if p_slip != 'default' and p_slip != 'def':
            overwrite['p_slip'] = p_slip
        overwrite['neglect_boarders'] = neglect_boarders
        self.mdp = OvercookedGridworld.from_layout_name(layout,**overwrite)
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon,time_cost=time_cost)
        obs_shape = self.mdp.get_lossless_encoding_vector_shape()
        master_config['trainer']['obs_shape'] = obs_shape

        self.LAYOUT = layout
        self.shared_rew = env_config['shared_rew']

        # Instantiate trainer configurations and Variables-------------
        self.init_sched(trainer_config['schedules'])
        self.feasible_action = FeasibleActionManager(self.env, enable=enable_feasible_actions)

        # Initialize Agent's policy and target networks ----------------
        loads = save_config['loads']
        self.rationality = agents_config['equilibrium']['rationality']
        self.test_rationality = agents_config['equilibrium']['rationality']
        self.cpt_params = agents_config['cpt']

        # model_config = agents_config['model']
        # model_config['rationality'] = agents_config['rationality'] # inherit rationality (minor change recommended)
        # Checkpointing/Saving utils ----------------
        self.checkpoint_score = -999
        self.checkpoint_mem = save_config['checkpoint_mem']
        # self.has_checkpointed = False
        self.train_rewards = deque(maxlen=self.checkpoint_mem)
        self.test_rewards = deque(maxlen=self.checkpoint_mem)
        self.fname_ext = save_config['fname_ext']
        self.save_dir = save_config['save_dir']
        self.wait_for_close = save_config['wait_for_close']
        self.auto_save = save_config['auto_save']

        # Load/Instantiate Model ------------------------------
        if loads == 'rational':
            rational_fname = f"{layout}_pslip{str(p_slip).replace('.', '')}__rational__"
            # self.model = model_object.from_file(obs_shape, n_actions, config,rational_fname)
            self.model = model_object.from_file(obs_shape, n_actions, agents_config, rational_fname)
        elif loads == '':
            self.model = model_object(obs_shape, n_actions, agents_config)
        elif loads.lower() == 'latest':
            loads = self.get_fname(with_ext=False,with_timestamp=False)  # get the model (no date)
            self.model = model_object.from_file(obs_shape, n_actions, agents_config, loads)
        else:
            self.model = model_object.from_file(obs_shape, n_actions, agents_config, loads)
            # raise ValueError(f"Invalid load option: {config['loads']}")




        # Initiate Logger and Managers ----------------
        self._init_logger_()
        self.enable_report = logger_config['enable_report']
        if self.enable_report: self.print_config(master_config)

    def _init_logger_(self):
        self.logger = RLLogger_V2(num_iters=self.ITERATIONS,wait_for_close = self.wait_for_close)

        self.logger.add_lineplot('test_reward', xlabel='', ylabel='$R_{test}$', filter_window=10, xtick=False)
        self.logger.add_lineplot('train_reward', xlabel='', ylabel='$R_{train}$', filter_window=50,xtick=False)
        self.logger.add_lineplot('loss', xlabel='iter', ylabel='$Loss$', filter_window=10)
        self.logger.add_checkpoint_watcher('test_reward', draw_on=['test_reward', 'train_reward','loss'], callback=self.checkpoint_callback)
        self.logger.add_settings(self.get_logger_display_data(self.master_config))


        # Create Watchers for training params
        self.rshape_scale = self.rshape_sched[0]  # initial rshape scale
        self.epsilon = self.epsilon_sched[0]  # initial epsilon
        self.iteration = 0
        def get_rshape():  return self.rshape_scale
        def get_epsilon(): return self.epsilon
        def get_prog(): return f'{round(self.iteration/self.ITERATIONS,2)*100}%'
        def get_qval_range():
            return self.model.qval_range# if hasattr(self.model, 'qval_range') else None
        self.logger.add_status('$\epsilon$', callback=get_epsilon)
        self.logger.add_status('$r_{s}$',callback=get_rshape) # reward shaping scale
        # self.logger.add_status('Prog', callback=get_prog)
        self.logger.add_status('$Q\in$', callback=get_qval_range)

        self.traj_visualizer = TrajectoryVisualizer(self.env)
        self.traj_heatmap = TrajectoryHeatmap(self.env)
        self.logger.add_button('Preview', callback=self.traj_visualizer.preview_qued_trajectory)
        self.logger.add_button('Heatmap', callback=self.traj_heatmap.preview)
        self.logger.add_button('Save ', callback=self.save)
        self.logger.add_toggle_button('wait_for_close', label='Wait For Close')
        # self.logger.add_checkbox('wait_for_close', label='Halt')
        # self.enable_report = self.logger_config['enable_report']

    def get_fname(self, with_ext=True, with_timestamp=True):
        if (self.cpt_params['b'] == 0
                and self.cpt_params['lam'] == 1.0
                and self.cpt_params['eta_p'] == 1.0
                and self.cpt_params['eta_n'] == 1.0
                and self.cpt_params['delta_p'] == 1.0
                and self.cpt_params['delta_n'] == 1.0):
            h = f"{self.LAYOUT}" \
                f"_pslip{str(self.mdp.p_slip).replace('.', '')}" \
                f"__rational"
        else:
            h = f"{self.LAYOUT}" \
                f"_pslip{str(self.mdp.p_slip).replace('.', '')}" \
                f"__b{str(self.cpt_params['b']).replace('.', '')}" \
                f"_lam{str(self.cpt_params['lam']).replace('.', '')}" \
                f"_etap{str(self.cpt_params['eta_p']).replace('.', '')}" \
                f"_etan{str(self.cpt_params['eta_n']).replace('.', '')}" \
                f"_deltap{str(self.cpt_params['delta_p']).replace('.', '')}" \
                f"_deltan{str(self.cpt_params['delta_n']).replace('.', '')}"

        if with_timestamp:
            h += f"__{self.timestamp}"
        if with_ext:
            h = self.fname_ext + h
        return h

    @property
    def fname(self):
        return self.get_fname()


    def get_logger_display_data(self,master_config):
        data = {}

        data['ALGORITHM'] = master_config['ALGORITHM']
        data['fname'] = self.fname

        data['ENVIRONMENT'] = '================================'
        data['layout'] = f"{master_config['env']['LAYOUT']}"+" $p_{slip}$ = " + f"{self.mdp.p_slip}"
        # data['p_slip'] = self.mdp.p_slip #master_config['env']['p_slip']
        data['time_cost'] = self.env.time_cost  # master_config['env']['p_slip']

        # data['shared_rew'] = master_config['env']['shared_rew']
        data['neglect boarder'] = master_config['env']['neglect_boarders']

        data['TRAINER'] = '####################################'
        data['ITERATIONS'] = master_config['trainer']['ITERATIONS']
        data['OBS Shape'] = master_config['trainer']['obs_shape']
        # data['warmup_transitions'] = master_config['trainer']['warmup_transitions']
        # data['N_tests'] = master_config['trainer']['N_tests']
        # data['test_interval'] = master_config['trainer']['test_interval']
        # data['shared_rew'] = master_config['trainer']['shared_rew']
        data['feasible_actions'] = master_config['trainer']['feasible_actions']
        data['Auto Save'] = master_config['save']['auto_save']

        # data['SCHEDULES'] = '================================'
        data['epsilon'] = list(master_config['trainer']['schedules']['epsilon_sched'].values())
        # data['random start'] = list(master_config['trainer']['schedules']['rand_start_sched'].values())
        data['rew shaping'] = list(master_config['trainer']['schedules']['rshape_sched'].values())

        data['AGENTS'] = '####################################'
        data['type'] = master_config['agents']['type']
        # data['device'] = master_config['agents']['model']['device']

        # data['rationality'] = master_config['agents']['equilibrium']['rationality']
        # data['lr'] = master_config['agents']['model']['lr']
        # data['gamma'] = master_config['agents']['model']['gamma']
        # data['tau'] = master_config['agents']['model']['tau']
        #
        data[''] = f"$\lambda$={master_config['agents']['equilibrium']['rationality']}\t" \
                   f"$\\alpha$={master_config['agents']['model']['lr']}\t" \
                   f"$\gamma$={master_config['agents']['model']['gamma']}\t" \
                   f"$\\tau$={master_config['agents']['model']['tau']}"\


        data['Mem Size'] = master_config['agents']['model']['replay_memory_size']
        data['Minibatch Size'] = master_config['agents']['model']['minibatch_size']
        data['NN Shape'] = f"{self.model.model.size_hidden_layers}" \
                           f"x{self.model.model.num_hidden_layers}" \
                           f" with {self.model.model.activation_function_name} activation"
                           # f" with {master_config['agents']['model']['activation']} activation"
        data['Clip Grad'] = master_config['agents']['model']['clip_grad']
        # data['CPT'] = master_config['agents']['cpt']
        data['CPT'] = " {" + f"$b$={master_config['agents']['cpt']['b']}, " \
                   f"$\ell$={master_config['agents']['cpt']['lam']}, " \
                   f"$\eta_p$={master_config['agents']['cpt']['eta_p']}, " \
                   f"$\eta_n$={master_config['agents']['cpt']['eta_n']}, " \
                   f"$\delta_p$={master_config['agents']['cpt']['delta_p']}, " \
                   f"$\delta_n$={master_config['agents']['cpt']['delta_n']}" + "}"
        return data

    def print_config(self,config):
        for key, val in config.items():
            print(f'{key}={val}')
    def init_sched(self,schedules,eps_decay = 1,rshape_decay=1):
        def exp_decay_sched(schedule,total_iterations):
            """ nonlinear time transformation where higher decay param ==> steeper decay"""
            START = schedule['start']
            END = schedule['end']
            DUR = schedule['duration']
            DECAY = schedule['decay']
            if DUR <= 1: # duration given in percent
                DUR = int(total_iterations * DUR)

            iters = np.arange(0, total_iterations)
            if START == END:  return np.ones(total_iterations) * START
            _sched = START * (END / START) ** ((iters/ DUR) ** (1 / DECAY))
            _sched = np.clip(_sched, END, None)
            return _sched

        def linear_decay_sched(schedule,total_iterations):
            START = schedule['start']
            END = schedule['end']
            DUR = schedule['duration']
            if DUR <= 1:  # duration given in percent
                DUR = int(total_iterations * DUR)

            _sched = np.hstack([np.linspace(START, END, DUR), END * np.ones(total_iterations - DUR)])
            return _sched

        # Scale Starting points
        schedules['rshape_sched']['start'] = schedules['rshape_sched']['start'] * rshape_decay
        schedules['epsilon_sched']['start'] = schedules['epsilon_sched']['start'] * eps_decay

        # Define schedules
        self.epsilon_sched = exp_decay_sched(schedules['epsilon_sched'], self.ITERATIONS)
        self.random_start_sched = exp_decay_sched(schedules['rand_start_sched'],self.ITERATIONS)
        self.rshape_sched = linear_decay_sched(schedules['rshape_sched'],self.ITERATIONS)

        # import matplotlib.pyplot as plt
        # plt.ioff()
        # plt.plot( self.epsilon_sched )
        # plt.xlabel('Iterations')
        # plt.ylabel('Epsilon')
        # plt.title('Epsilon Schedule')
        # plt.grid()
        # plt.show()


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
                self.training_rollout(it,rationality=self.rationality,
                                      epsilon = self.epsilon_sched[it],
                                      rshape_scale= self.rshape_sched[it],
                                      p_rand_start=self.random_start_sched[it])

            if it>1: self.model.scheduler.step() # updates learning rate scheduler
            self.model.update_target()  # performs soft update of target network
            self.logger.end_iteration()

            # slips = rollout_info['onion_slips'] + rollout_info['dish_slips'] + rollout_info['soup_slips']
            risks = rollout_info['onion_risked'] + rollout_info['dish_risked'] + rollout_info['soup_risked']
            handoffs = rollout_info['onion_handoff'] + rollout_info['dish_handoff'] + rollout_info['soup_handoff']
            if self.enable_report:
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
                      # f"| rat:{round(self.rationality_sched[it], 3)}"
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
                    # if not self.has_checkpointed:
                    #     self.traj_visualizer.que_trajectory(state_history)
                    #     self.traj_heatmap.que_trajectory(state_history)

                # Checkpointing ----------------------
                self.test_rewards.append(np.mean(test_rewards))  # for checkpointing
                self.train_rewards.append(np.mean(train_rewards))  # for checkpointing
                self.checkpoint(it,state_history)
                # if self.checkpoint(it):  # check if should checkpoint
                #     self.traj_visualizer.que_trajectory(state_history) # load preview of checkpointed trajectory
                #     self.traj_heatmap.que_trajectory(state_history)
                # Logging ----------------------
                self.logger.log(test_reward=[it, np.mean(test_rewards)],
                           train_reward=[it, np.mean(train_rewards)],
                           loss=[it, np.mean(train_losses)])
                self.logger.draw()
                if self.enable_report:
                    print(f"\nTest: | nTests= {self.N_tests} "
                          f"| Ave Reward = {np.mean(test_rewards)} "
                          f"| Ave Shaped Reward = {np.mean(test_shaped_rewards)}"
                          # f"\n{action_history}\n"#, f"{aprob_history[0]}\n"
                          )

                train_rewards = []
                train_losses = []
        self.logger.wait_for_close(enable=self.wait_for_close)
        # self.logger.wait_for_close(enable=True)
        if self.auto_save: self.save()

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
            obs = self.mdp.get_lossless_encoding_vector_astensor(self.env.state,device=self.device).unsqueeze(0)
            feasible_JAs = self.feasible_action.get_feasible_joint_actions(self.env.state,as_joint_idx=True)
            joint_action, joint_action_idx, action_probs = self.model.choose_joint_action(obs,
                                                                                          epsilon=epsilon,
                                                                                          feasible_JAs = feasible_JAs)
            next_state_prospects = self.mdp.one_step_lookahead(self.env.state.deepcopy(),
                                                               joint_action=Action.ALL_JOINT_ACTIONS[joint_action_idx],
                                                               as_tensor=True, device=self.device)
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
    def test_rollout(self,rationality,epsilon=0,rshape_scale=1,get_info = False):

        if get_info:
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

            }
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
            obs = self.mdp.get_lossless_encoding_vector_astensor(self.env.state,device=self.device).unsqueeze(0)
            joint_action, joint_action_idx, action_probs = self.model.choose_joint_action(obs, epsilon=epsilon)
            next_state, reward, done, info = self.env.step(joint_action)

            # Track reward traces
            test_reward += reward
            test_shaped_reward += rshape_scale*np.mean(info["shaped_r_by_agent"])*np.ones(2)

            # Track state-action history
            action_history.append(joint_action_idx)
            aprob_history.append(action_probs)
            state_history.append(next_state.deepcopy())

            if get_info:
                for key in rollout_info.keys():
                    if key not in ['mean_loss']:
                        rollout_info[key] += np.array(info['mdp_info']['event_infos'][key])

            if done:  break
            self.env.state = next_state
        self.model.model.train()
        self.model.target.train()
        if get_info: return test_reward, test_shaped_reward, state_history, action_history, aprob_history, rollout_info
        else: return test_reward, test_shaped_reward, state_history, action_history, aprob_history

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
    def checkpoint_callback(self):
        self.model.update_checkpoint()
        self.traj_visualizer.que_trajectory(self.state_history)
        self.traj_heatmap.que_trajectory(self.state_history)


    def checkpoint(self,it, state_history): #TODO: Delete since LoggerV2

        score = np.mean(self.test_rewards)
        if score > self.checkpoint_score:
            if self.enable_report:
                print(f'\nCheckpointing model at iteration {it} with score {score}...\n')
            self.model.update_checkpoint()
            self.logger.update_checkpiont_line(it)
            self.checkpoint_score = score
            self.has_checkpointed = True

            self.traj_visualizer.que_trajectory(state_history)
            self.traj_heatmap.que_trajectory(state_history)
            return True
        #########################################
        # if len(self.train_rewards) == self.checkpoint_mem:
        #     ave_train = np.mean(self.train_rewards)
        #     ave_test = np.mean(self.test_rewards)
        #     # score = (ave_train + ave_test)/2
        #     score = ave_test
        #     if score > self.min_checkpoint_score and score > self.checkpoint_score:
        #         print(f'\nCheckpointing model at iteration {it} with score {score}...\n')
        #         self.model.update_checkpoint()
        #         self.logger.update_checkpiont_line(it)
        #         self.checkpoint_score = score
        #         self.has_checkpointed = True
        #         return True
        # return False


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

    def save(self,*args, save_model=True, save_fig=True):
        # find saved models absolute dir -------------
        print(f'\n\nSaving model to {self.fname}...')
        dir = get_absolute_save_dir(path=self.save_dir)

        if save_model:
            torch.save(self.model.checkpoint_model.state_dict(), dir + f"{self.fname}.pt")

        if save_fig:
            self.logger.save_fig(f"{dir}{self.fname}.png")
        print(f'finished\n\n')




if __name__ == "__main__":
    raise NotImplementedError