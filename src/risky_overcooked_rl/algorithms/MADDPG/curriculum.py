from risky_overcooked_py.mdp.overcooked_mdp import ObjectState,SoupState
from risky_overcooked_rl.algorithms.MADDPG.trainer import CPTTrainer
import numpy as np
from risky_overcooked_py.mdp.actions import Action
from collections import deque
import math
import scipy.stats as stats
import torch
from risky_overcooked_rl.utils.state_utils import invert_obs, invert_joint_action, invert_prospect

debug = False


class CirriculumTrainer(CPTTrainer):
    def __init__(self,master_config):
        super().__init__(master_config)

        # Parse Curriculum Config ---------
        curriculum_config = master_config.curriculum.__dict__
        self.schedule_decay = curriculum_config['schedule_decay']
        # self.schedules = master_config['trainer']['schedules']
        self.curriculum = Curriculum(self.env,curriculum_config)


    def run(self):
        sparse_rewards_buffer = []
        shaped_rewards_buffer = []

        self.warmup()
        # Main training Loop
        for epi in range(self.num_episodes):
            self.logger.start_iteration()
            self.logger.spin()


            self.training_iteration = epi
            cit = self.curriculum.iteration

            prog = epi / self.num_episodes
            self.step_exploration(step=cit)


            ##########################################################
            # Training Step ##########################################
            # Perform Rollout

            # TRAINING ROLLOUT ##################################
            episode_sparse_reward = 0
            episode_shaped_reward = 0
            self.env.reset()
            obs = self.mdp.get_lossless_encoding_vector_astensor(self.env.state, device=self.device)

            for t in range(self.env.horizon):
                self.logger.spin()  # prevents plot from freezing

                # Act
                agent_observations = torch.vstack([obs, invert_obs(obs)])
                agent_actions = self.agent.act(agent_observations, sample=True)
                joint_action_idx = Action.INDEX_TO_ACTION_INDEX_PAIRS.index(tuple(agent_actions))
                joint_action = self.joint_action_space[joint_action_idx]

                # Step state-action
                next_state_prospects = self.mdp.one_step_lookahead(self.env.state.deepcopy(),
                                                                   joint_action=Action.ALL_JOINT_ACTIONS[
                                                                       joint_action_idx], as_tensor=True,
                                                                   device=self.device)
                next_state, sparse_rewards, done, info = self.env.step(joint_action)
                next_obs = self.mdp.get_lossless_encoding_vector_astensor(self.env.state, self.device)
                shaped_rewards = info['shaped_r_by_agent']

                # Log and store in memory and update
                episode_sparse_reward += sparse_rewards
                episode_shaped_reward += np.mean(shaped_rewards)
                total_rewards = np.mean(sparse_rewards) + np.array(shaped_rewards).reshape(-1, 1)
                self.replay_buffer.double_push(state=obs,
                                               action=joint_action_idx,
                                               rewards=total_rewards,
                                               next_prospects=next_state_prospects,
                                               done=done)
                self.agent.update(self.replay_buffer)


                # Close step
                obs = next_obs
                if done:
                    print(f'[{round(prog * 100, 1)}% | epi: {epi}] train/episode_reward',
                          [episode_sparse_reward, episode_shaped_reward])
                    break

            # Log episode stats
            sparse_rewards_buffer.append(episode_sparse_reward)
            shaped_rewards_buffer.append(episode_shaped_reward)

            # Check Curriculum Advancement ================
            rew = episode_sparse_reward + episode_shaped_reward
            next_cirriculum = self.curriculum.step_cirriculum(rew)




            if epi % self.eval_episode_freq == 0:
                self.curriculum.eval('on')
                ave_eval_reward,state_history = self.evaluate()
                self.logger.log(
                    test_reward=[epi, np.mean(ave_eval_reward)],
                    train_reward=[epi, np.mean(sparse_rewards_buffer) + np.mean(shaped_rewards_buffer)],
                )
                self.checkpoint_score_buffer.append(np.mean(ave_eval_reward))
                self.checkpoint(epi, state_history)
                self.logger.draw()
                sparse_rewards_buffer = []
                shaped_rewards_buffer = []
                self.curriculum.eval('off')


            # Close Iteration ########################################
            # Check if model training is failing and halt -----------
            if self.curriculum.is_failing(epi, self.num_episodes):
                print(f"Model is failing to learn at Curriculum {self.curriculum.name}. Ending training...")
                # self.save(save_model=self.curriculum.save_model_on_fail, save_fig=self.curriculum.save_fig_on_fail)
                break

            self.logger.end_iteration()


            ####### END TRAINING ON LOGGER CLOSED ########
            if self.logger.is_closed:
                print(f"Logger Closed at iteration {epi}. Ending training...")
                break

        self.logger.wait_for_close(enable=True)
        # self.logger.wait_for_close(enable=self.wait_for_close)
        # self.logger.close_plots()
        # if self.auto_save and not self.curriculum.is_failing(it, self.ITERATIONS):
        #     self.save(save_model=True, save_fig=True)



class Curriculum:
    def __init__(self, env, curriculum_config):

        # Parse Curriculum Config ----------------
        self.min_iterations_per_cirriculum = curriculum_config['min_iter']
        self.reward_buffer = deque(maxlen=curriculum_config['curriculum_mem'])

        # Initiate variables and Get meta information ---------------------
        self.iteration = 0  # iteration for this cirriculum
        self.default_params = {'n_onions': 3, 'cook_time': 20}
        self.iteration = 0 #iteration for this cirriculum
        self.current_cirriculum = 0

        self.env = env
        self.mdp = env.mdp
        self.layout = self.mdp.layout_name

        # Set up curriculum advancment thresholds ------------
        time_cost = self.env.time_cost
        timecost_offset = self.env.horizon*time_cost
        soup_reward = 20
        self.cirriculum_step_threshs = {}
        for key, num_soups in curriculum_config['subtask_goals'].__dict__.items():
            self.cirriculum_step_threshs[key] = soup_reward*num_soups + timecost_offset
        # self.cirriculum_step_threshs = {
        #     'deliver_soup': 80 + self.env.horizon*time_cost,
        #     'pick_up_soup': 80 + self.env.horizon*time_cost,
        #     'pick_up_dish': 70 + self.env.horizon*time_cost,
        #     'wait_to_cook': 50 + self.env.horizon*time_cost,
        #     'deliver_onion3': 50 + self.env.horizon*time_cost,
        #     'pick_up_onion3': 50 + self.env.horizon*time_cost,
        #     'deliver_onion2': 40 + self.env.horizon*time_cost,
        #     'pick_up_onion2': 40 + self.env.horizon*time_cost,
        #     'deliver_onion1': 40 + self.env.horizon*time_cost,
        #     'full_task': 999
        # }
        self.cirriculums = list(self.cirriculum_step_threshs.keys())
        self.name = self.cirriculums[self.current_cirriculum]

        # Set up failure detection thresholds (used for paramter searching)
        self.failure_thresh = curriculum_config['failure_checks'].__dict__
        self.save_fig_on_fail = curriculum_config['failure_checks'].__dict__['save_fig']
        self.save_model_on_fail = curriculum_config['failure_checks'].__dict__['save_model']

    def is_failing(self,training_iter, total_iter):
        """ checks if model learning is failing to progress efficiently"""
        training_prog = training_iter/total_iter
        if self.failure_thresh['enable']:
            return (training_prog > self.failure_thresh[self.name])
        else: return False

    def step_cirriculum(self, reward):
        self.reward_buffer.append(reward)
        reward_thresh  = self.cirriculum_step_threshs[self.cirriculums[self.current_cirriculum]]
        if (
                np.mean(self.reward_buffer) >= reward_thresh
                and self.current_cirriculum < len(self.cirriculums) -1
                and self.iteration > self.min_iterations_per_cirriculum
        ):
            self.current_cirriculum += 1
            self.iteration = 0
            self.name = self.cirriculums[self.current_cirriculum]
            return True
        self.iteration += 1
        return False

    def is_early_stopping(self,state,reward,info):
        """
        Goal of each cirriculum is to deliver 1 soup except for full_task
        """
        #TODO: This may not be valid. Better early stopping criteria? Maybe number of soups in play?
        #TODO: Change to when each subtask is complete?

        i = self.current_cirriculum
        if not self.cirriculums[i] == 'full_task' :
            # if reward == 20: return True # reward for delivering soup
            # # elif len(state.all_objects_list) ==0: return True # lost all objects
            if np.any(info['mdp_info']['event_infos']['soup_delivery']):  return True # delivering soup
        return False

    def pdf_curriculum_sample(self,curriculum_step,interpolate=False):
        n_curr = len(self.cirriculums) - 1
        mu = curriculum_step
        variance = 0.5
        max_deviation = 3
        sigma = math.sqrt(variance)

        if interpolate and self.cirriculums[curriculum_step] != 'full_task':
            reward_thresh = self.cirriculum_step_threshs[self.cirriculums[self.current_cirriculum]]
            prog = np.mean(self.reward_buffer)/reward_thresh
            mu += prog


        # x = np.linspace(mu - max_deviation*sigma, mu + max_deviation*sigma, 100)
        x = np.linspace(min(0, mu - max_deviation * sigma), min(n_curr, mu + max_deviation * sigma), 100)
        p_samples = stats.norm.pdf(x, mu, sigma)
        p_samples = p_samples / np.sum(p_samples)
        xi = np.random.choice(x, p=p_samples)
        return xi

    def sample_cirriculum_state(self, rand_start_chance = 0.9, sample_decay =0.5, pdf_sample=False):
        # i = self.current_cirriculum
        if pdf_sample:
            self.pdf_curriculum_sample(self.current_cirriculum)
        else:
            pi = [sample_decay**(self.current_cirriculum-i) for i in range(self.current_cirriculum+1)]
            i = np.random.choice(np.arange(self.current_cirriculum+1), p=np.array(pi)/np.sum(pi))

        state = self.add_random_start_loc()

        if self.cirriculums[i] == 'full_task':
            state = self.env.state # undo random start loc

        elif self.cirriculums[i] == 'deliver_onion1':
            """
            Start pot with 0 onion & one player with onion            
            """
            possibilities = [["onion", None], [None, "onion"]]
            held_objs = possibilities[np.random.randint(len(possibilities))]
            state = self.add_held_objs(state, held_objs)

        elif self.cirriculums[i] == 'pick_up_onion2':
            """
            Start pot with 1 onion            
            """
            n_onions = 1
            onion_quants = np.eye(self.mdp.num_pots, dtype=int)[np.random.choice(self.mdp.num_pots)] * n_onions
            state = self.add_onions_to_pots(state, onion_quants)

        elif self.cirriculums[i] == 'deliver_onion2':
            """
            Start pot with 1 onion & one player with onion
             - other pots unfilled since filling other pot before first is suboptimal
            """
            n_onions = 1
            onion_quants = np.eye(self.mdp.num_pots,dtype=int)[np.random.choice(self.mdp.num_pots)] * n_onions
            state = self.add_onions_to_pots(state, onion_quants)

            possibilities = [["onion", None], [None, "onion"]]
            held_objs = possibilities[np.random.randint(len(possibilities))]
            state = self.add_held_objs(state, held_objs)

        elif self.cirriculums[i] == 'pick_up_onion3':
            """
            Start with one pot filled with 2 onions
            - other pots unfilled since filling other pot before first is suboptimal
            """
            n_onions = 2
            onion_quants = np.eye(self.mdp.num_pots,dtype=int)[np.random.choice(self.mdp.num_pots)] * n_onions
            state = self.add_onions_to_pots(state, onion_quants)

        elif self.cirriculums[i] == 'deliver_onion3':
            """
            Start with one pot filled with 2 onions & one player with onion
            - other pots unfilled since filling other pot before first is suboptimal
            """
            possibilities = [["onion", None], [None, "onion"]]
            held_objs = possibilities[np.random.randint(len(possibilities))]
            state = self.add_held_objs(state, held_objs)

            n_onions = 2
            onion_quants = np.eye(self.mdp.num_pots,dtype=int)[np.random.choice(self.mdp.num_pots)] * n_onions
            state = self.add_onions_to_pots(state, onion_quants)

        elif self.cirriculums[i] == 'wait_to_cook':
            """
            Initiate pot with three onions loaded and waiting to cook 
            - other pots not filled since filling other pot before first starts cooking is suboptimal
            """
            # TODO: random num onions for other pots?
            n_onions = 3
            onion_quants = np.eye(self.mdp.num_pots,dtype=int)[np.random.choice(self.mdp.num_pots)] * n_onions
            state = self.add_onions_to_pots(state, onion_quants, cooking_tick= 0)

        elif self.cirriculums[i] == 'pick_up_dish':
            """
            Initiate pot with soup finished cooking
            - other pots partially filled = likely since already waited for soup to cook
            """
            # n_onions = 3
            # onion_quants = np.eye(self.mdp.num_pots)[np.random.choice(self.mdp.num_pots)] * n_onions
            # state = self.add_onions_to_pots(state, onion_quants, cooking_tick=20)  # finished soup is cooked
            n_onions = 3
            pots = np.eye(self.mdp.num_pots)[np.random.choice(self.mdp.num_pots)]  # which pot is filled
            onion_quants = [n_onions if pot == 1 else np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
                            for pot in pots]
            state = self.add_onions_to_pots(state, onion_quants, cooking_tick=20)  # finished soup is cooked

        elif self.cirriculums[i] == 'pick_up_soup':
            """
            Initiate pot with finished soup and [P1 or P2] holding plate
            - more than one soup finished cooking is unlikely
            - other pots partially filled = likely since already waited for soup to cook
            - both holding plate is unnecessary
            """
            # Decide who is holding dish and other is holding rand object
            possibilities = [["dish", np.random.choice([None, "onion"], p=[0.75, 0.25])],
                             [np.random.choice([None, "onion"], p=[0.75, 0.25]), "dish"]]
            held_objs = possibilities[np.random.randint(len(possibilities))]
            state = self.add_held_objs(state, held_objs)

            # Sample how many onions to put in each pot (i.e. one is full rest are empty)
            # n_onions = 3
            # onion_quants = np.eye(self.mdp.num_pots)[np.random.choice(self.mdp.num_pots)]*n_onions
            # state = self.add_onions_to_pots(state, onion_quants, cooking_tick=20)  # finished soup is cooked
            n_onions = 3
            pots = np.eye(self.mdp.num_pots)[np.random.choice(self.mdp.num_pots)] # which pot is filled
            onion_quants = [n_onions if pot == 1 else np.random.choice([0,1,2],p=[0.7,0.2,0.1])
                            for pot in pots]
            state = self.add_onions_to_pots(state, onion_quants, cooking_tick=20)  # finished soup is cooked

        elif self.cirriculums[i] == 'deliver_soup':
            """
            Init [P1 or P2] with held soup 
            - both holding soup is unnecessary 
            - randomize other player held object
            - one pot likely empty and others likely partial filled
            """
            possibilities = [["soup",np.random.choice([None,"onion"], p=[0.75, 0.25])],
                             [np.random.choice([None,"onion"], p=[0.75, 0.25]), "soup"]]
            held_soups = possibilities[np.random.randint(len(possibilities))]
            state = self.add_held_objs(state, held_soups)
        else:
            raise ValueError(f"Invalid cirriculum mode '{self.cirriculums[i]}'")
        self.env.state = state
        return state

    def eval(self,status):
        # if status.lower() == 'on': self.set_cirriculum(**self.default_params)
        # elif status.lower() == 'off':  self.set_cirriculum(**self.cirriculums[self.current_cirriculum])
        # else: raise ValueError(f"Invalid curriculum test mode status '{status}'. Use 'on' or 'off'")
        pass

    def add_random_start_loc(self):
        random_state = self.mdp.get_random_start_state_fn(random_start_pos=True, rnd_obj_prob_thresh=0.0)()
        return random_state

    def add_onions_to_pots(self, state, onion_quants,cooking_tick=None):
        pots = self.mdp.get_pot_states(state)["empty"]
        assert len(onion_quants)==len(pots), "Number of pots must match number of onion quantities"
        for n_onion,pot_loc in zip(onion_quants,pots):
            if cooking_tick is None:
                cooking_tick = np.random.randint(0, 20) if n_onion == 3 else -1
            if n_onion > 0:
                state.objects[pot_loc] = SoupState.get_soup(
                    pot_loc,
                    num_onions=n_onion,
                    num_tomatoes=0,
                    cooking_tick=cooking_tick,
                )
        return state

    def add_held_objs(self, state, objs):
        # For each player, add a random object with prob rnd_obj_prob_thresh
        for obj,player in zip(objs,state.players):
            if obj is not None:
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

    def add_held_obj(self, player, obj):

        if obj == "soup":
            player.set_object(SoupState.get_soup(player.position, num_onions=3, num_tomatoes=0, finished=True))
        else:
            player.set_object(ObjectState(obj, player.position))
        return player
# def main():
#     config = {
#         'ALGORITHM': 'Boltzmann_QRE-DDQN-OSA',
#         'Date': datetime.now().strftime("%m_%d_%Y-%H_%M"),
#
#         # Env Params ----------------
#         # 'LAYOUT': "risky_coordination_ring",
#         # 'LAYOUT': "risky_multipath",
#         # 'LAYOUT': "forced_coordination",
#         # 'LAYOUT': "forced_coordination_sanity_check",
#         'LAYOUT': "sanity_check",
#         'HORIZON': 200,
#         'ITERATIONS': 30_000,
#         'AGENT': None,  # name of agent object (computed dynamically)
#         "obs_shape": None,  # computed dynamically based on layout
#         "p_slip": 0.5,
#
#         # Learning Params ----------------
#         "rand_start_sched": [0.1, 0.1, 10_000],  # percentage of ITERATIONS with random start states
#         'epsilon_sched': [0.9, 0.15, 5000],  # epsilon-greedy range (start,end)
#         'rshape_sched': [1, 0, 10_000],  # rationality level range (start,end)
#         'rationality_sched': [5, 5, 10_000],
#         'lr_sched': [1e-4, 1e-4, 1],
#         'gamma': 0.97,  # discount factor
#         'tau': 0.01,  # soft update weight of target network
#         "num_hidden_layers": 5,  # MLP params
#         "size_hidden_layers": 256,  # 32,      # MLP params
#         "device": self.device,
#         "minibatch_size": 256,  # size of mini-batches
#         "replay_memory_size": 20_000,  # size of replay memory
#         'clip_grad': 100,
#         'monte_carlo': False,
#         'note': 'Cirriculum OSA',
#     }
#
#     # config['LAYOUT'] = 'forced_coordination'; config['rand_start_sched'] = [0,0,1]
#     # config['LAYOUT'] = 'risky_coordination_ring'; config['tau'] = 0.005
#     # config['LAYOUT'] = 'risky_multipath'
#     # config['LAYOUT'] = 'forced_coordination_sanity_check'; config['rand_start_sched'] = [0,0,1]
#     CirriculumTrainer(SelfPlay_QRE_OSA, config).run()

# class CirriculumTrainer(Trainer):
#     def __init__(self,model_object,custom_config):
#         super().__init__(model_object,custom_config)
#         self.curriculum = Curriculum(self.env,time_cost=custom_config['time_cost'])
#         self.schedule_decay = 0.7
#
#     def run(self):
#         train_rewards = []
#         train_losses = []
#         # Main training Loop
#         for it in range(self.ITERATIONS):
#             cit = self.curriculum.iteration
#             self.logger.start_iteration()
#             self.logger.spin()
#
#             ##########################################################
#             # Training Step ##########################################
#             # Perform Rollout
#
#             cum_reward, cum_shaped_rewards, rollout_info = \
#                 self.curriculum_rollout(cit,
#                                         rationality=self.rationality_sched[cit],
#                                         epsilon=self.epsilon_sched[cit],
#                                         rshape_scale=self.rshape_sched[cit],
#                                         p_rand_start=self.random_start_sched[cit])
#
#             if it > 1: self.model.scheduler.step()  # updates learning rate scheduler
#             self.model.update_target()  # performs soft update of target network
#
#
#             next_cirriculum = self.curriculum.step_cirriculum(cum_reward)
#             if next_cirriculum:
#                 self.init_sched(self.config, eps_decay=self.schedule_decay, rshape_decay=self.schedule_decay)
#             risks = rollout_info['onion_risked'] + rollout_info['dish_risked'] + rollout_info['soup_risked']
#             handoffs = rollout_info['onion_handoff'] + rollout_info['dish_handoff'] + rollout_info['soup_handoff']
#             if self.enable_report:
#                 print(f"[it:{it}"
#                       f" {self.curriculum.name}:{self.curriculum.current_cirriculum}-{cit}]"
#                       f"[R:{round(cum_reward, 3)} "
#                       f" Rshape:{np.round(cum_shaped_rewards, 3)} "
#                       f" L:{round(rollout_info['mean_loss'], 3)} ]"
#                       # f"| slips:{slips} "
#                       f"[ risks:{risks} "
#                       f" handoffs:{handoffs} ]"
#                       f" |"
#                       f"| mem:{len(self.model._memory)} "
#                       f"| rshape:{round(self.rshape_sched[cit], 3)} "
#                       f"| rat:{round(self.rationality_sched[cit], 3)}"
#                       f"| eps:{round(self.epsilon_sched[cit], 3)} "
#                       f"| LR={round(self.model.optimizer.param_groups[0]['lr'], 4)}"
#                       f"| rstart={round(self.random_start_sched[cit], 3)}"
#                       )
#
#             train_rewards.append(cum_reward + cum_shaped_rewards)
#             train_losses.append(rollout_info['mean_loss'])
#
#             # Testing Step ##########################################
#             time4test = (it % self.test_interval == 0)
#             if time4test and rollout_info['mean_loss']>0:
#                 self.curriculum.eval('on')
#
#                 # Rollout test episodes ----------------------
#                 test_rewards = []
#                 test_shaped_rewards = []
#                 for test in range(self.N_tests):
#                     test_reward, test_shaped_reward, state_history, action_history, aprob_history =\
#                         self.test_rollout(rationality=self.test_rationality)
#                     test_rewards.append(test_reward)
#                     test_shaped_rewards.append(test_shaped_reward)
#                     # if not self.has_checkpointed:
#                     #     self.traj_visualizer.que_trajectory(state_history)
#                     #     self.traj_heatmap.que_trajectory(state_history)
#
#                 # Checkpointing ----------------------
#                 self.test_rewards.append(np.mean(test_rewards))  # for checkpointing
#                 self.train_rewards.append(np.mean(train_rewards))  # for checkpointing
#                 self.checkpoint(it,state_history)
#                 # if self.checkpoint(it):  # check if should checkpoint
#                 #     self.traj_visualizer.que_trajectory(state_history)  # load preview of checkpointed trajectory
#                 #     self.traj_heatmap.que_trajectory(state_history)
#
#                 # Logging ----------------------
#                 self.logger.log(test_reward=[it, np.mean(test_rewards)],
#                                 train_reward=[it, np.mean(train_rewards)],
#                                 loss=[it, np.mean(train_losses)])
#                 self.logger.draw()
#
#                 if self.enable_report:
#                     print(f"\nTest: | nTests= {self.N_tests} "
#                           f"| Ave Reward = {np.mean(test_rewards)} "
#                           f"| Ave Shaped Reward = {np.mean(test_shaped_rewards)}"
#                           # f"\n{action_history}\n"#, f"{aprob_history[0]}\n"
#                           )
#
#                 train_rewards = []
#                 train_losses = []
#                 self.curriculum.eval('off')
#             self.logger.end_iteration()
#         # self.logger.wait_for_close(enable=True)
#         self.logger.wait_for_close(enable=self.wait_for_close)
#         if self.auto_save: self.save()
#         self.logger.close_plots()
#     def curriculum_rollout(self, it, rationality,epsilon,rshape_scale,p_rand_start=0):
#         self.model.rationality = rationality
#         self.env.reset()
#         self.env.state = self.curriculum.sample_cirriculum_state()
#
#         losses = []
#         cum_reward = 0
#         cum_shaped_reward = np.zeros(2)
#
#         rollout_info = {
#             'onion_risked': np.zeros([1, 2]),
#             'onion_pickup': np.zeros([1, 2]),
#             'onion_drop': np.zeros([1, 2]),
#             'onion_handoff': np.zeros([1, 2]),
#
#             'dish_risked': np.zeros([1, 2]),
#             'dish_pickup': np.zeros([1, 2]),
#             'dish_drop': np.zeros([1, 2]),
#             'dish_handoff': np.zeros([1, 2]),
#
#             'soup_risked': np.zeros([1, 2]),
#             'soup_pickup': np.zeros([1, 2]),
#             'soup_delivery': np.zeros([1, 2]),
#             'soup_drop': np.zeros([1, 2]),
#             'soup_handoff': np.zeros([1, 2]),
#
#             'soup_slip': np.zeros([1, 2]),
#             'onion_slip': np.zeros([1, 2]),
#             'dish_slip': np.zeros([1, 2]),
#
#             'mean_loss': 0
#         }
#
#         for t in range(self.env.horizon+1):#itertools.count():
#             old_state = self.env.state.deepcopy()
#             obs = self.mdp.get_lossless_encoding_vector_astensor(self.env.state, device=device).unsqueeze(0)
#             feasible_JAs = self.feasible_action.get_feasible_joint_actions(self.env.state, as_joint_idx=True)
#             joint_action, joint_action_idx, action_probs = self.model.choose_joint_action(obs,
#                                                                                           epsilon=epsilon,
#                                                                                           feasible_JAs=feasible_JAs)
#             next_state, reward, done, info = self.env.step(joint_action, get_mdp_info=True)
#             next_state_prospects = self.mdp.one_step_lookahead(old_state, # must be called after step....
#                                                                joint_action=Action.ALL_JOINT_ACTIONS[joint_action_idx],
#                                                                as_tensor=True, device=device)
#
#             for key in rollout_info.keys():
#                 if not key == 'mean_loss':
#                     rollout_info[key] += np.array(info['mdp_info']['event_infos'][key])
#
#
#             # Track reward traces
#             shaped_rewards = rshape_scale * np.array(info["shaped_r_by_agent"])
#             if self.shared_rew: shaped_rewards = np.mean(shaped_rewards) * np.ones(2)
#             total_rewards = np.array([reward + shaped_rewards]).flatten()
#             cum_reward += reward
#             cum_shaped_reward += shaped_rewards
#
#             # Store in memory ----------------
#             self.model._memory.double_push(state=obs,
#                                           action=joint_action_idx,
#                                           rewards=total_rewards,
#                                           next_prospects=next_state_prospects,
#                                           done=done)
#             # Update model ----------------
#             loss = self.model.update()
#             if loss is not None: losses.append(loss)
#
#             # Terminate episode
#             if done: break
#             elif self.curriculum.is_early_stopping(self.env.state,reward,info):  # Cirriculum Complete, reset state in this episode
#             # elif info['mdp_info']['soup_delivery']:  # Cirriculum Complete, reset state in this episode
#                 self.env.state = self.curriculum.sample_cirriculum_state()
#                 # print(f"Early Stopping at t={t}")
#             else: self.env.state = next_state
#
#         rollout_info['mean_loss'] = np.mean(losses)
#
#         return cum_reward, cum_shaped_reward, rollout_info
#
#     # def checkpoint(self,it):
#     #     if len(self.train_rewards) == self.checkpoint_mem:
#     #         # ave_train = np.mean(self.train_rewards)
#     #         ave_test = np.mean(self.test_rewards)
#     #         # score = (ave_train + ave_test)/2
#     #         score = ave_test
#     #         if score > self.min_checkpoint_score and score >= self.checkpoint_score:
#     #             print(f'\nCheckpointing model at iteration {it} with score {score}...\n')
#     #             self.model.update_checkpoint()
#     #             self.logger.update_checkpiont_line(it)
#     #             # empty buffer to delay next checkpoint
#     #             self.train_rewards = deque(maxlen=self.checkpoint_mem)
#     #             self.test_rewards = deque(maxlen=self.checkpoint_mem)
#     #             self.has_checkpointed = True
#     #             self.checkpoint_score = score
#     #             return True
#     #     return False
#
#
#
# class Curriculum:
#     def __init__(self, env, time_cost=0):
#         self.env = env
#         self.mdp = env.mdp
#         self.layout = self.mdp.layout_name
#         # self.reward_thresh = reward_thresh
#         self.reward_buffer = deque(maxlen=10)
#         self.iteration = 0 #iteration for this cirriculum
#         self.min_iterations_per_cirriculum = 100
#         self.default_params = {'n_onions': 3, 'cook_time': 20}
#
#         self.current_cirriculum = 0
#         # self.cirriculum_step_threshs = {
#         #     'deliver_soup': 80,
#         #     'pick_up_soup': 80,
#         #     'pick_up_dish': 60,
#         #     'wait_to_cook': 40,
#         #     #'deliver_onion3': 40,
#         #     'pick_up_onion3': 40,
#         #     #'deliver_onion2': 40,
#         #     'pick_up_onion2': 40,
#         #     #'deliver_onion1': 40,
#         #     'full_task': 40
#         # }
#         self.cirriculum_step_threshs = {
#             'deliver_soup': 80 + self.env.horizon*time_cost,
#             'pick_up_soup': 80 + self.env.horizon*time_cost,
#             'pick_up_dish': 70 + self.env.horizon*time_cost,
#             'wait_to_cook': 50 + self.env.horizon*time_cost,
#             'deliver_onion3': 50 + self.env.horizon*time_cost,
#             'pick_up_onion3': 50 + self.env.horizon*time_cost,
#             'deliver_onion2': 40 + self.env.horizon*time_cost,
#             'pick_up_onion2': 40 + self.env.horizon*time_cost,
#             'deliver_onion1': 40 + self.env.horizon*time_cost,
#             'full_task': 999
#         }
#
#         self.cirriculums = list(self.cirriculum_step_threshs.keys())
#         self.name = self.cirriculums[self.current_cirriculum]
#
#
#
#     def step_cirriculum(self, reward):
#         self.reward_buffer.append(reward)
#         reward_thresh  = self.cirriculum_step_threshs[self.cirriculums[self.current_cirriculum]]
#         if (
#                 np.mean(self.reward_buffer) >= reward_thresh
#                 and self.current_cirriculum < len(self.cirriculums) -1
#                 and self.iteration > self.min_iterations_per_cirriculum
#         ):
#             self.current_cirriculum += 1
#             self.iteration = 0
#             self.name = self.cirriculums[self.current_cirriculum]
#             return True
#         self.iteration += 1
#         return False
#
#     def is_early_stopping(self,state,reward,info):
#         """
#         Goal of each cirriculum is to deliver 1 soup except for full_task
#         """
#         #TODO: This may not be valid. Better early stopping criteria? Maybe number of soups in play?
#         #TODO: Change to when each subtask is complete?
#
#         i = self.current_cirriculum
#         if not self.cirriculums[i] == 'full_task' :
#             # if reward == 20: return True # reward for delivering soup
#             # # elif len(state.all_objects_list) ==0: return True # lost all objects
#             if np.any(info['mdp_info']['event_infos']['soup_delivery']):  return True # delivering soup
#         return False
#
#     def pdf_curriculum_sample(self,curriculum_step,interpolate=False):
#         n_curr = len(self.cirriculums) - 1
#         mu = curriculum_step
#         variance = 0.5
#         max_deviation = 3
#         sigma = math.sqrt(variance)
#
#         if interpolate and self.cirriculums[curriculum_step] != 'full_task':
#             reward_thresh = self.cirriculum_step_threshs[self.cirriculums[self.current_cirriculum]]
#             prog = np.mean(self.reward_buffer)/reward_thresh
#             mu += prog
#
#
#         # x = np.linspace(mu - max_deviation*sigma, mu + max_deviation*sigma, 100)
#         x = np.linspace(min(0, mu - max_deviation * sigma), min(n_curr, mu + max_deviation * sigma), 100)
#         p_samples = stats.norm.pdf(x, mu, sigma)
#         p_samples = p_samples / np.sum(p_samples)
#         xi = np.random.choice(x, p=p_samples)
#         return xi
#
#     def sample_cirriculum_state(self, rand_start_chance = 0.9, sample_decay =0.5, pdf_sample=False):
#         # i = self.current_cirriculum
#         if pdf_sample:
#             self.pdf_curriculum_sample(self.current_cirriculum)
#         else:
#             pi = [sample_decay**(self.current_cirriculum-i) for i in range(self.current_cirriculum+1)]
#             i = np.random.choice(np.arange(self.current_cirriculum+1), p=np.array(pi)/np.sum(pi))
#
#         state = self.add_random_start_loc()
#
#         if self.cirriculums[i] == 'full_task':
#             state = self.env.state # undo random start loc
#
#         elif self.cirriculums[i] == 'deliver_onion1':
#             """
#             Start pot with 0 onion & one player with onion
#             """
#             possibilities = [["onion", None], [None, "onion"]]
#             held_objs = possibilities[np.random.randint(len(possibilities))]
#             state = self.add_held_objs(state, held_objs)
#
#         elif self.cirriculums[i] == 'pick_up_onion2':
#             """
#             Start pot with 1 onion
#             """
#             n_onions = 1
#             onion_quants = np.eye(self.mdp.num_pots, dtype=int)[np.random.choice(self.mdp.num_pots)] * n_onions
#             state = self.add_onions_to_pots(state, onion_quants)
#
#         elif self.cirriculums[i] == 'deliver_onion2':
#             """
#             Start pot with 1 onion & one player with onion
#              - other pots unfilled since filling other pot before first is suboptimal
#             """
#             n_onions = 1
#             onion_quants = np.eye(self.mdp.num_pots,dtype=int)[np.random.choice(self.mdp.num_pots)] * n_onions
#             state = self.add_onions_to_pots(state, onion_quants)
#
#             possibilities = [["onion", None], [None, "onion"]]
#             held_objs = possibilities[np.random.randint(len(possibilities))]
#             state = self.add_held_objs(state, held_objs)
#
#         elif self.cirriculums[i] == 'pick_up_onion3':
#             """
#             Start with one pot filled with 2 onions
#             - other pots unfilled since filling other pot before first is suboptimal
#             """
#             n_onions = 2
#             onion_quants = np.eye(self.mdp.num_pots,dtype=int)[np.random.choice(self.mdp.num_pots)] * n_onions
#             state = self.add_onions_to_pots(state, onion_quants)
#
#         elif self.cirriculums[i] == 'deliver_onion3':
#             """
#             Start with one pot filled with 2 onions & one player with onion
#             - other pots unfilled since filling other pot before first is suboptimal
#             """
#             possibilities = [["onion", None], [None, "onion"]]
#             held_objs = possibilities[np.random.randint(len(possibilities))]
#             state = self.add_held_objs(state, held_objs)
#
#             n_onions = 2
#             onion_quants = np.eye(self.mdp.num_pots,dtype=int)[np.random.choice(self.mdp.num_pots)] * n_onions
#             state = self.add_onions_to_pots(state, onion_quants)
#
#         elif self.cirriculums[i] == 'wait_to_cook':
#             """
#             Initiate pot with three onions loaded and waiting to cook
#             - other pots not filled since filling other pot before first starts cooking is suboptimal
#             """
#             # TODO: random num onions for other pots?
#             n_onions = 3
#             onion_quants = np.eye(self.mdp.num_pots,dtype=int)[np.random.choice(self.mdp.num_pots)] * n_onions
#             state = self.add_onions_to_pots(state, onion_quants, cooking_tick= 0)
#
#         elif self.cirriculums[i] == 'pick_up_dish':
#             """
#             Initiate pot with soup finished cooking
#             - other pots partially filled = likely since already waited for soup to cook
#             """
#             # n_onions = 3
#             # onion_quants = np.eye(self.mdp.num_pots)[np.random.choice(self.mdp.num_pots)] * n_onions
#             # state = self.add_onions_to_pots(state, onion_quants, cooking_tick=20)  # finished soup is cooked
#             n_onions = 3
#             pots = np.eye(self.mdp.num_pots)[np.random.choice(self.mdp.num_pots)]  # which pot is filled
#             onion_quants = [n_onions if pot == 1 else np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
#                             for pot in pots]
#             state = self.add_onions_to_pots(state, onion_quants, cooking_tick=20)  # finished soup is cooked
#
#         elif self.cirriculums[i] == 'pick_up_soup':
#             """
#             Initiate pot with finished soup and [P1 or P2] holding plate
#             - more than one soup finished cooking is unlikely
#             - other pots partially filled = likely since already waited for soup to cook
#             - both holding plate is unnecessary
#             """
#             # Decide who is holding dish and other is holding rand object
#             possibilities = [["dish", np.random.choice([None, "onion"], p=[0.75, 0.25])],
#                              [np.random.choice([None, "onion"], p=[0.75, 0.25]), "dish"]]
#             held_objs = possibilities[np.random.randint(len(possibilities))]
#             state = self.add_held_objs(state, held_objs)
#
#             # Sample how many onions to put in each pot (i.e. one is full rest are empty)
#             # n_onions = 3
#             # onion_quants = np.eye(self.mdp.num_pots)[np.random.choice(self.mdp.num_pots)]*n_onions
#             # state = self.add_onions_to_pots(state, onion_quants, cooking_tick=20)  # finished soup is cooked
#             n_onions = 3
#             pots = np.eye(self.mdp.num_pots)[np.random.choice(self.mdp.num_pots)] # which pot is filled
#             onion_quants = [n_onions if pot == 1 else np.random.choice([0,1,2],p=[0.7,0.2,0.1])
#                             for pot in pots]
#             state = self.add_onions_to_pots(state, onion_quants, cooking_tick=20)  # finished soup is cooked
#
#         elif self.cirriculums[i] == 'deliver_soup':
#             """
#             Init [P1 or P2] with held soup
#             - both holding soup is unnecessary
#             - randomize other player held object
#             - one pot likely empty and others likely partial filled
#             """
#             possibilities = [["soup",np.random.choice([None,"onion"], p=[0.75, 0.25])],
#                              [np.random.choice([None,"onion"], p=[0.75, 0.25]), "soup"]]
#             held_soups = possibilities[np.random.randint(len(possibilities))]
#             state = self.add_held_objs(state, held_soups)
#         else:
#             raise ValueError(f"Invalid cirriculum mode '{self.cirriculums[i]}'")
#         self.env.state = state
#         return state
#
#     def eval(self,status):
#         # if status.lower() == 'on': self.set_cirriculum(**self.default_params)
#         # elif status.lower() == 'off':  self.set_cirriculum(**self.cirriculums[self.current_cirriculum])
#         # else: raise ValueError(f"Invalid curriculum test mode status '{status}'. Use 'on' or 'off'")
#         pass
#
#     def random_start_state(self):
#         state = self.add_random_start_loc()
#         state = self.add_random_start_pot_state(state)
#         state = self.add_random_held_obj(state)
#         # state = self.add_random_counter_state(state)
#         return state
#
#     def add_random_start_loc(self):
#         random_state = self.mdp.get_random_start_state_fn(random_start_pos=True, rnd_obj_prob_thresh=0.0)()
#         return random_state
#
#     def add_onions_to_pots(self, state, onion_quants,cooking_tick=None):
#         pots = self.mdp.get_pot_states(state)["empty"]
#         assert len(onion_quants)==len(pots), "Number of pots must match number of onion quantities"
#         for n_onion,pot_loc in zip(onion_quants,pots):
#             if cooking_tick is None:
#                 cooking_tick = np.random.randint(0, 20) if n_onion == 3 else -1
#             if n_onion > 0:
#                 state.objects[pot_loc] = SoupState.get_soup(
#                     pot_loc,
#                     num_onions=n_onion,
#                     num_tomatoes=0,
#                     cooking_tick=cooking_tick,
#                 )
#         return state
#
#     def add_held_objs(self, state, objs):
#         # For each player, add a random object with prob rnd_obj_prob_thresh
#         for obj,player in zip(objs,state.players):
#             if obj is not None:
#                 self.add_held_obj(player, obj)
#         return state
#
#     def add_random_counter_state(self, state, rnd_obj_prob_thresh=0.025):
#         counters = self.mdp.reachable_counters
#         for counter_loc in counters:
#             p = np.random.rand()
#             if p < rnd_obj_prob_thresh:
#                 obj = np.random.choice(["onion", "dish", "soup"], p=[0.6, 0.2, 0.2])
#                 if obj == "soup":
#                     state.add_object(SoupState.get_soup(counter_loc, num_onions=3, num_tomatoes=0, finished=True))
#                 else:
#                     state.add_object(ObjectState(obj, counter_loc))
#         return state
#
#     def add_held_obj(self, player, obj):
#
#         if obj == "soup":
#             player.set_object(SoupState.get_soup(player.position, num_onions=3, num_tomatoes=0, finished=True))
#         else:
#             player.set_object(ObjectState(obj, player.position))
#         return player
# def main():
#     config = {
#         'ALGORITHM': 'Boltzmann_QRE-DDQN-OSA',
#         'Date': datetime.now().strftime("%m_%d_%Y-%H_%M"),
#
#         # Env Params ----------------
#         # 'LAYOUT': "risky_coordination_ring",
#         # 'LAYOUT': "risky_multipath",
#         # 'LAYOUT': "forced_coordination",
#         # 'LAYOUT': "forced_coordination_sanity_check",
#         'LAYOUT': "sanity_check",
#         'HORIZON': 200,
#         'ITERATIONS': 30_000,
#         'AGENT': None,  # name of agent object (computed dynamically)
#         "obs_shape": None,  # computed dynamically based on layout
#         "p_slip": 0.5,
#
#         # Learning Params ----------------
#         "rand_start_sched": [0.1, 0.1, 10_000],  # percentage of ITERATIONS with random start states
#         'epsilon_sched': [0.9, 0.15, 5000],  # epsilon-greedy range (start,end)
#         'rshape_sched': [1, 0, 10_000],  # rationality level range (start,end)
#         'rationality_sched': [5, 5, 10_000],
#         'lr_sched': [1e-4, 1e-4, 1],
#         'gamma': 0.97,  # discount factor
#         'tau': 0.01,  # soft update weight of target network
#         "num_hidden_layers": 5,  # MLP params
#         "size_hidden_layers": 256,  # 32,      # MLP params
#         "device": device,
#         "minibatch_size": 256,  # size of mini-batches
#         "replay_memory_size": 20_000,  # size of replay memory
#         'clip_grad': 100,
#         'monte_carlo': False,
#         'note': 'Cirriculum OSA',
#     }
#
#     # config['LAYOUT'] = 'forced_coordination'; config['rand_start_sched'] = [0,0,1]
#     # config['LAYOUT'] = 'risky_coordination_ring'; config['tau'] = 0.005
#     # config['LAYOUT'] = 'risky_multipath'
#     # config['LAYOUT'] = 'forced_coordination_sanity_check'; config['rand_start_sched'] = [0,0,1]
#     CirriculumTrainer(SelfPlay_QRE_OSA, config).run()

# if __name__ == '__main__':
#     main()
