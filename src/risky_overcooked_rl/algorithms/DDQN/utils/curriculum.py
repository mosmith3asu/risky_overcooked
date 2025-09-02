from risky_overcooked_py.mdp.overcooked_mdp import ObjectState,SoupState, Direction
from risky_overcooked_rl.algorithms.DDQN.utils.trainer import Trainer
import numpy as np
from risky_overcooked_py.mdp.actions import Action
from collections import deque
import math
import scipy.stats as stats
debug = False

from risky_overcooked_py.planning.planners import MotionPlanner
class CurriculumTrainer(Trainer):
    def __init__(self,model_object,master_config):
        super().__init__(model_object,master_config)

        # Parse Curriculum Config ---------
        curriculum_config = master_config['trainer']['curriculum']
        self.schedule_decay = curriculum_config.get('schedule_decay',1.0)
        self.schedules = master_config['trainer']['schedules']
        # self.reset_on_service = master_config['trainer']['curriculum'].get('reset_on_service', True)
        self.curriculum = Curriculum(self.env,curriculum_config)


    def run(self):
        self.logger.enable_checkpointing(True)
        pending_tests = []  # to keep track of async test processes

        # Main training Loop
        for it in range(self.ITERATIONS + self.EXTRA_ITERATIONS):
            train_rewards = []
            train_losses = []

            self.training_iteration = it
            cit = self.curriculum.iteration
            self.logger.loop_iteration()
            self.logger.spin()

            ##########################################################
            # Training Step ##########################################
            self.iteration = it # for logging callbacks
            # self.epsilon = self.epsilon_sched[cit] if cit < len(self.epsilon_sched) else self.epsilon_sched[-1] # for logging callbacks
            # self.rshape_scale = self.rshape_sched[cit] if cit < len(self.rshape_sched) else self.rshape_sched[-1] # for logging callbacks
            # self.random_start = self.random_start_sched[cit] if cit < len(self.random_start_sched) else self.random_start_sched[-1] # for logging callbacks
            self.epsilon = self.epsilon_sched[min(cit,len(self.epsilon_sched)-1)]  # for logging callbacks
            self.rshape_scale = self.rshape_sched[min(cit,len(self.rshape_sched)-1)]
            self.random_start = self.random_start_sched[min(cit,len(self.rshape_sched)-1)]


            cum_reward, cum_shaped_rewards, rollout_info = \
                self.curriculum_rollout(cit,
                                        rationality=self.rationality,
                                        epsilon=self.epsilon,
                                        rshape_scale=self.rshape_scale,
                                        p_rand_start=self.random_start)

            train_rewards.append(cum_reward + cum_shaped_rewards)
            train_losses.append(rollout_info['mean_loss'])

            did_update = (rollout_info['mean_loss']!=0)
            self.risk_taken.append(rollout_info['risks_taken'])

            if did_update:
                self.model.update_target()  # performs soft update of target network

                # TODO: Made edits here. CRITICAL
                step_val = rollout_info['reward_latest_curriculum']
                if step_val is not None:
                    is_next_curriculum = self.curriculum.step_curriculum(step_val)
                else:
                    is_next_curriculum = False
                # is_next_curriculum = self.curriculum.step_curriculum(cum_reward)

                if is_next_curriculum:
                    self.init_sched(self.schedules, eps_decay=self.schedule_decay, rshape_decay=self.schedule_decay)

                # if self.curriculum.curr_curriculum_name in self.curriculum.curriculums[-3:-1]:
                #     self.logger.enable_checkpointing(True)

            # Report ###########################################
            if self.enable_report:
                self.report(it,cum_reward,cum_shaped_rewards,rollout_info)

            self.logger.log(train_reward=[it, cum_reward + np.mean(cum_shaped_rewards)],
                            loss=[it, rollout_info['mean_loss']],
                            # eps=self.epsilon_sched[cit]
                            )

            # Testing Step ##########################################
            # if it % self.test_interval == 0 and it > 0:
            #     # Rollout async test episodes ----------------------
            #     proc, queue = self.async_test_start(self.N_tests)
            #     pending_tests.append((proc, queue, it))  # add to pending tests
            #
            # all_test_it, test_results, pending_tests = self.async_test_check(pending_tests)
            # for test_it, res in zip(all_test_it,test_results):
            #     test_reward = res['test_reward']
            #     # test_shaped_reward = res['test_shaped_reward']
            #     # state_history = res['state_history']
            #     # action_history = res['action_history']
            #     # aprob_history = res['aprob_history']
            #
            #     self.test_rewards.append(test_reward)  # for checkpointing
            #     self.train_rewards.append(np.mean(train_rewards))  # for checkpointing
            #
            #     # self.checkpoint(it, state_history)
            #
            #     self.logger.log(test_reward=[test_it, np.mean(self.test_rewards)])
            #     self.logger.draw()

            time4test = (it % self.test_interval == 0 and it>0)
            if time4test:
                # Rollout test episodes ----------------------
                test_rewards = []
                test_shaped_rewards = []
                state_historys = []
                for test in range(self.N_tests):
                    test_reward, test_shaped_reward, state_history, action_history, aprob_history =\
                        self.test_rollout(rationality=self.test_rationality)
                    test_rewards.append(test_reward)
                    test_shaped_rewards.append(test_shaped_reward)
                    state_historys += state_history

                self.state_history = state_historys
                # self.test_rewards.append(np.mean(test_rewards))
                # self.train_rewards.append(np.mean(test_rewards))

                # Logging ----------------------
                self.logger.log(test_reward=[it, np.mean(test_rewards)])
                self.logger.draw()
                if self.enable_report:
                    print(f"\nTest: | nTests= {self.N_tests} "
                          f"| Ave Reward = {np.mean(test_rewards)} "
                          f"| Ave Shaped Reward = {np.mean(test_shaped_rewards)}"
                          )


            # Close Iteration ########################################
            # Check if model training is failing and halt -----------
            if self.curriculum.is_failing(it, self.ITERATIONS):
                print(f"Model is failing to learn at Curriculum {self.curriculum.name}. Ending training...")
                self.save(save_model=self.curriculum.save_model_on_fail, save_fig=self.curriculum.save_fig_on_fail)
                break

            ####### END TRAINING ON LOGGER CLOSED ########
            if self.logger.is_closed:
                print(f"Logger Closed at iteration {it}. Ending training...")
                break

        self.logger.halt()
        self.logger.close_plots()
        if self.auto_save and not self.curriculum.is_failing(it, self.ITERATIONS):
            self.save(save_model=True, save_fig=True)

    def curriculum_rollout(self, it, rationality,epsilon,rshape_scale,p_rand_start=0):
        init_epsilon = epsilon
        self.model.rationality = rationality
        self.env.reset()


        state,sampled_curriculum = self.curriculum.sample_curriculum_state()
        # is_latest_curriculum = sampled_curriculum == self.curriculum.curr_curriculum_name
        self.curriculum.subtask_iters[sampled_curriculum] += 1

        t_latest_curriculum = 0  # total timesteps for this curriculum
        reward_latest_curriculum = 0  # total sparse reward for this curriculum

        if not self.curriculum.is_latest_curriculum:
            eps_gain = self.curriculum.completed_epsilon
            epsilon = max(eps_gain * init_epsilon, self.epsilon_sched[-1]) if eps_gain is not None else epsilon

        self.env.state = state

        losses = []
        cum_sparse_reward = 0
        cum_shaped_reward = np.zeros(2)

        rollout_info = {
            'onion_risked': np.zeros([1, 2]),
            'onion_pickup': np.zeros([1, 2]),
            'onion_drop': np.zeros([1, 2]),
            'onion_handoff': np.zeros([1, 2]),

            'dish_risked': np.zeros([1, 2]),
            'dish_pickup': np.zeros([1, 2]),
            'dish_drop': np.zeros([1, 2]),
            'dish_handoff': np.zeros([1, 2]),

            'soup_risked': np.zeros([1, 2]),
            'soup_pickup': np.zeros([1, 2]),
            'soup_delivery': np.zeros([1, 2]),
            'soup_drop': np.zeros([1, 2]),
            'soup_handoff': np.zeros([1, 2]),

            'soup_slip': np.zeros([1, 2]),
            'onion_slip': np.zeros([1, 2]),
            'dish_slip': np.zeros([1, 2]),

            'mean_loss': 0
        }


        for t in range(self.env.horizon+1):#itertools.count():
            if t%5==0: self.logger.spin()

            old_state = self.env.state.deepcopy()#.freeze()
            # obs = self.mdp.get_lossless_encoding_vector_astensor(self.env.state, device=self.device).unsqueeze(0)
            obs = self.mdp.get_lossless_encoding_vector_astensor(old_state, device=self.device).unsqueeze(0)

            # feasible_JAs = self.feasible_action.get_feasible_joint_actions(self.env.state, as_joint_idx=True)
            feasible_JAs = self.feasible_action.get_feasible_joint_actions(old_state, as_joint_idx=True)

            joint_action, joint_action_idx, action_probs = \
                self.model.choose_joint_action(obs,epsilon=epsilon, feasible_JAs=feasible_JAs)

            next_state, sparse_reward, done, info = self.env.step(joint_action, get_mdp_info=True)

            next_state_prospects = self.mdp.one_step_lookahead(old_state, # must be called after step....
                                                               joint_action=Action.ALL_JOINT_ACTIONS[joint_action_idx],
                                                               as_tensor=True, device=self.device)

            for key in rollout_info.keys():
                if not key == 'mean_loss':
                    rollout_info[key] += np.array(info['mdp_info']['event_infos'][key])


            # Track reward traces
            shaped_rewards = rshape_scale * np.array(info["shaped_r_by_agent"])
            cum_sparse_reward += sparse_reward
            cum_shaped_reward += shaped_rewards

            reward_latest_curriculum += sparse_reward if self.curriculum.is_latest_curriculum else 0 #TODO: ReMOVE?
            t_latest_curriculum += 1 if self.curriculum.is_latest_curriculum else 0

            # Store in memory ----------------
            self.model._memory.double_push(state=obs,
                                          action=joint_action_idx,
                                          rewards=np.array([sparse_reward + shaped_rewards]).flatten(),
                                          next_prospects=next_state_prospects,
                                          done=done)
            # Update model ----------------
            if len(self.model._memory) > self.warmup_transitions:
                loss = self.model.update()
                if loss is not None:
                    losses.append(loss)
            else: losses.append(0)

            # Terminate episode ##################################
            if done:
                break
            elif self.curriculum.is_reset_needed(self.env.state,sparse_reward,info):
                # Curriculum Complete, reset state in this episode
                # Sample new curriculum state
                self.env.state , sampled_curriculum= self.curriculum.sample_curriculum_state()
                is_latest_curriculum = sampled_curriculum == self.curriculum.curr_curriculum_name
                epsilon = init_epsilon if is_latest_curriculum else self.curriculum.completed_epsilon * init_epsilon
                epsilon = max(epsilon, self.epsilon_sched[-1])
            else:
                self.env.state = next_state

        rollout_info['mean_loss'] = np.mean(losses)

        if t_latest_curriculum>0:
            latest_reward_scale = self.env.horizon/t_latest_curriculum
            rollout_info['reward_latest_curriculum'] = reward_latest_curriculum * latest_reward_scale
        else:
            rollout_info['reward_latest_curriculum'] = None

        rollout_info['risks_taken'] = np.sum([np.sum(rollout_info[key]) for key in
                            ['dish_risked','onion_risked','soup_risked']] )


        return cum_sparse_reward, cum_shaped_reward, rollout_info

    def report(self,it,cum_reward,cum_shaped_rewards,rollout_info):
        risks = rollout_info['onion_risked'] + rollout_info['dish_risked'] + rollout_info['soup_risked']
        handoffs = rollout_info['onion_handoff'] + rollout_info['dish_handoff'] + rollout_info['soup_handoff']
        cit = self.curriculum.iteration

        print(f"[it:{it}"
              f" {self.curriculum.curr_curriculum_name}:-{cit}]"
              f"[R:{round(cum_reward, 3)} "
              f" Rshape:{np.round(cum_shaped_rewards, 3)} "
              f" L:{round(rollout_info['mean_loss'], 3)} ]"
              # f"| slips:{slips} "
              f"[ risks:{risks} "
              f" handoffs:{handoffs} ]"
              f" |"
              f"| mem:{len(self.model._memory)} "
              f"| rshape:{round(self.rshape_sched[cit], 3)} "
              # f"| rat:{round(self.rationality, 3)}"
              f"| eps:{round(self.epsilon_sched[cit], 3)} "
              f"| LR={round(self.model.optimizer.param_groups[0]['lr'], 4)}"
              f"| rstart={round(self.random_start_sched[cit], 3)}"
              )



class Curriculum:
    def __init__(self, env, curriculum_config, **kwargs):
        # Initiate variables and Get meta information ---------------------
        self.iteration = 0  # iteration for this curriculum
        self.current_curriculum = kwargs.pop('current_curriculum',0)
        self.env = env
        self.mdp = env.mdp


        # Parse Curriculum Config ----------------
        self.sampling = curriculum_config['sampling']
        assert self.sampling in ['pdf', 'uniform', 'decay','decay2'], 'unknown curriculum sampling'
        self.sampling_decay = curriculum_config['sampling_decay']
        self.min_iterations = curriculum_config['min_iter']
        self.completed_epsilon = curriculum_config['completed_epsilon']
        self.reward_buffer = deque(maxlen=curriculum_config['curriculum_mem'])
        self.save_fig_on_fail = curriculum_config['failure_checks']['save_fig']
        self.save_model_on_fail = curriculum_config['failure_checks']['save_model']
        add_rshape_thresh = curriculum_config['add_rshape_goals']

        self.reset_on_service = curriculum_config['reset_on_service']

        self.failure_thresh = curriculum_config['failure_checks']
        self.subtask_goals = curriculum_config['subtask_goals']
        assert set(self.subtask_goals.keys()).issubset(set(self.failure_thresh.keys())), \
            "Each subtask goal must have a corresponding failure threshold defined "

        self.subtask_iters = {}
        for key in self.subtask_goals.keys():
            self.subtask_iters[key] = 0

        # Set up curriculum advancment thresholds ------------
        time_cost = self.env.time_cost
        timecost_offset = self.env.horizon*time_cost
        soup_reward = 20

        self.curriculum_step_threshs = {}
        for key, num_soups in self.subtask_goals.items():
            shaped_rewards = 0
            # add shaped rewards to goal
            if add_rshape_thresh:
                if key in ['pick_up_soup']:
                    shaped_rewards +=  self.mdp.reward_shaping_params['SOUP_PICKUP_REWARD']
                if key in ['pick_up_dish']:
                    shaped_rewards += self.mdp.reward_shaping_params['SOUP_PICKUP_REWARD']
                    shaped_rewards += self.mdp.reward_shaping_params['DISH_PICKUP_REWARD']
                if 'deliver_onion' in key or 'pick_up_onion' in key:
                    n = 4-int(key[-1])
                    shaped_rewards += n * self.mdp.reward_shaping_params['PLACEMENT_IN_POT_REW']
                    shaped_rewards += self.mdp.reward_shaping_params['SOUP_PICKUP_REWARD']
                    shaped_rewards += self.mdp.reward_shaping_params['DISH_PICKUP_REWARD']
                elif 'full_task' in key:
                    n=3
                    shaped_rewards += n * self.mdp.reward_shaping_params['PLACEMENT_IN_POT_REW']
                    shaped_rewards += self.mdp.reward_shaping_params['SOUP_PICKUP_REWARD']
                    shaped_rewards += self.mdp.reward_shaping_params['DISH_PICKUP_REWARD']

            self.curriculum_step_threshs[key] = (soup_reward + shaped_rewards) * num_soups + timecost_offset

        self.curriculums = list(self.curriculum_step_threshs.keys())


        # Use Motion Planner to get distance to goal ------------
        self.mp = MotionPlanner(self.mdp)
        self.navigation_dist_hash = {}
        finite_dists = np.where(np.isfinite(self.mp.graph_problem.distance_matrix))
        self._max_dist2goal = np.max(self.mp.graph_problem.distance_matrix[finite_dists])
        self.start_near_goal_iters = kwargs.pop('start_near_goal_iters', 1)


    #########################################################################################################
    # Curriculum Sampling ###################################################################################
    def step_curriculum(self, reward):
        self.reward_buffer.append(reward)
        reward_thresh = self.curriculum_step_threshs[self.curriculums[self.current_curriculum]]
        if (
                np.mean(self.reward_buffer) >= reward_thresh
                and self.current_curriculum < len(self.curriculums) - 1
                and self.iteration > self.min_iterations
        ):
            self.current_curriculum += 1
            self.iteration = 0
            return True
        self.iteration += 1
        return False

    def pdf_curriculum_sample(self,curriculum_step,interpolate=False):
        n_curr = len(self.curriculums) - 1
        mu = curriculum_step
        variance = 0.5
        max_deviation = 3
        sigma = math.sqrt(variance)

        if interpolate and self.curriculums[curriculum_step] != 'full_task':
            reward_thresh = self.curriculum_step_threshs[self.curriculums[self.current_curriculum]]
            prog = np.mean(self.reward_buffer)/reward_thresh
            mu += prog

        x = np.linspace(min(0, mu - max_deviation * sigma), min(n_curr, mu + max_deviation * sigma), 100)
        p_samples = stats.norm.pdf(x, mu, sigma)
        p_samples = p_samples / np.sum(p_samples)
        xi = np.random.choice(x, p=p_samples)
        return xi


    def deliver_onion(self,N, state,p=0.5):

        situation = np.random.choice([0, 1], p=[p, 1-p])
        if situation == 0:
            """Other onions already in pot, onion N on the way"""
            n_onions = N-1
            held_objs = [["onion", None], [None, "onion"]]
            held_objs = held_objs[np.random.randint(len(held_objs))]  # randomly select one of the held objects to be an onion
        else:
            """onion 2 on the way, need to deliver anouther"""
            n_onions = max(0,N-2)
            held_objs = ["onion", "onion"]

        onions_in_play = n_onions + np.sum([1  for obj in held_objs if obj=='onion'])
        # assert onions_in_play == N, f"(DELIVER) Total number of onions in play ({onions_in_play})mismatched with specified {N} onions"
        assert n_onions >= 0 and n_onions <= 3, "Number of onions must be between 0 and 3"

        state = self.add_held_objs(state, held_objs)
        state = self.add_onions_to_pots(state, n_onions=n_onions)
        return state

    def pickup_onion(self,N,state,p=0.5):
        situation = np.random.choice([0, 1], p=[p, 1-p])
        if situation == 0:
            """Onion N-1 already in pot, need to pick up onion"""
            n_onions = N-1
            held_objs = [None, None]
        else:
            """onion N-1 on the way, need to pick up onion N"""
            n_onions = max(0,N-2)
            held_objs = [["onion", None], [None, "onion"]]

        # onions_in_play = n_onions + np.sum([1 for obj in held_objs if obj == 'onion'])
        # assert onions_in_play == N-1, f"Total number of onions in play mismatched with specified {N} onions"
        assert n_onions >= 0 and n_onions <= 3, "Number of onions must be between 0 and 3"

        state = self.add_held_objs(state, held_objs)
        state = self.add_onions_to_pots(state, n_onions=n_onions)
        return state

    def sample_curriculum(self):
        if self.sampling == 'pdf':
            i = self.pdf_curriculum_sample(self.current_curriculum)
        elif self.sampling == 'uniform':
            i = np.random.randint(0, len(self.curriculums))
            self.current_curriculum = -1
        elif self.sampling == 'decay':
            # only sample previous curriculums
            pi = [self.sampling_decay**(self.current_curriculum-i) for i in range(self.current_curriculum+1)]
            i = np.random.choice(np.arange(self.current_curriculum+1), p=np.array(pi)/np.sum(pi))
        elif self.sampling == 'decay2':
            # also sample future curriculum
            pi = [self.sampling_decay ** (self.current_curriculum - i) for i in range(self.current_curriculum + 1)]
            pi = 0.95*np.array(pi) / np.sum(pi)

            rem_curric = len(self.curriculums) - (self.current_curriculum+1)
            pj = 0.05*np.ones(rem_curric)/rem_curric # probability of sampling future curriculum

            p= np.concatenate((pi, pj))
            p = p/np.sum(p)
            i = np.random.choice(len(self.curriculums), p=p)

        self.sampled_curriculum = self.curriculums[i]
        return self.sampled_curriculum

    def sample_curriculum_state(self):

        self.sampled_curriculum = self.sample_curriculum()

        state = self.init_random_state()

        if self.sampled_curriculum == 'full_task':
            state = self.env.state # undo random start loc

        elif 'deliver_onion' in self.sampled_curriculum:
            N = int(self.sampled_curriculum[-1])  # number of onions to deliver
            state = self.deliver_onion(N, state)

        elif 'pick_up_onion' in self.sampled_curriculum:
            N = int(self.sampled_curriculum[-1])  # number of onions to deliver
            state = self.deliver_onion(N, state)

        elif self.sampled_curriculum == 'pick_up_dish':
            """
            Initiate pot with soup finished cooking
            - other pots partially filled = likely since already waited for soup to cook
            """
            situation = np.random.choice([0, 1], p=[0.5, 0.5])
            if situation == 0:
                """3 onion already in pot, mobody has onion"""
                n_onions = 3
                held_objs = [None, None]
                cooking_tick = np.random.randint(0,20)
            else:
                "onion 3 on its way, need to pick up dish"
                n_onions = 2
                held_objs = [["onion", None],[None,"onion"]]
                cooking_tick = None

            state = self.add_held_objs(state, held_objs)
            state = self.add_onions_to_pots(state, n_onions=n_onions, cooking_tick=cooking_tick)  # finished soup is cooked

        elif self.sampled_curriculum == 'pick_up_soup':
            """
            Initiate pot with finished soup and [P1 or P2] holding plate
            - more than one soup finished cooking is unlikely
            - other pots partially filled = likely since already waited for soup to cook
            - both holding plate is unnecessary
            """
            # Decide who is holding dish and other is holding rand object
            n_onions = 3
            partner_item = np.random.choice([None, "onion"], p=[0.9, 0.1])
            held_objs = [["dish", partner_item], [partner_item , "dish"]]
            cooking_tick = 20

            state = self.add_held_objs(state, held_objs)
            state = self.add_onions_to_pots(state, n_onions = n_onions, cooking_tick=cooking_tick)  # finished soup is cooked


        elif self.sampled_curriculum == 'deliver_soup':
            """
            Init [P1 or P2] with held soup 
            - both holding soup is unnecessary 
            - randomize other player held object
            - one pot likely empty and others likely partial filled
            """
            n_onions = 0
            partner_item = np.random.choice([None, "onion"], p=[0.8, 0.2])
            held_objs = [["soup",partner_item],  [ partner_item, "soup"]]
            cooking_tick = None

            state = self.add_held_objs(state, held_objs)
            state = self.add_onions_to_pots(state, n_onions=n_onions, cooking_tick=cooking_tick)  # finished soup is cooked

            for player in state.players:
                if player.has_object() and player.get_object().name=='soup':
                    self.assign_dist2goal_start_loc(state, player, 'pot',distance=1)  # start near pot
                    break

        else:
            raise ValueError(f"Invalid curriculum mode '{self.sampled_curriculum}'")
        return state, self.sampled_curriculum

    def eval(self,status):
        # if status.lower() == 'on': self.set_curriculum(**self.default_params)
        # elif status.lower() == 'off':  self.set_curriculum(**self.curriculums[self.current_curriculum])
        # else: raise ValueError(f"Invalid curriculum test mode status '{status}'. Use 'on' or 'off'")
        pass

    ########################################################################################################
    # START STATE FUNCTIONS ################################################################################

    def init_random_state(self):
        """generates random state locations and held objects"""
        random_state = self.mdp.get_random_start_state_fn(random_start_pos=True, rnd_obj_prob_thresh=0.0)()
        return random_state

    def assign_random_start_loc(self, player, exclude=()):
        """ Samples a random start location for the player, excluding specified positions."""
        valid_pos = self.mdp.get_valid_player_positions()
        for pos in exclude:
            valid_pos.remove(pos)
        valid_pos = valid_pos[np.random.randint(len(valid_pos))]
        player.update_pos_and_or(valid_pos, player.orientation)

    def assign_dist2goal_start_loc(self, state, player, goal,distance):
        """ Samples random start location for player, near (depending on iter) goal asset location."""
        if distance >= self._max_dist2goal:
            return # no need to add start loc if distance is max

        # Sekect goal asset location
        if goal == "serving":  # start at pot
            locs = self.mdp.get_serving_locations()
        elif goal == "onion":  # start at onion dispenser
            locs = self.mdp.get_onion_dispenser_locations()
        elif goal == "dish":  # start at dish dispenser
            locs = self.mdp.get_dish_dispenser_locations()
        elif goal == "pot":  # start at pot
            # locs = self.mdp.get_pot_locations()
            locs = None
            pot_states = self.mdp.get_pot_states(state)
            for key in ['ready', 'cooking', '2_items', '1_items', "empty"]:
                if len(pot_states[key]) > 0:
                    # print('sampled pot state:', key, 'with', len(pot_states[key]), 'pots')
                    locs = pot_states[key]
                    break
            assert locs is not None, "No pots found in state to start at"

        else:
            raise ValueError(f"Invalid goal type '{goal}' for logical start location")
        asset_loc = locs[np.random.randint(len(locs))]
        goal_pos = None
        checked_terrains = []
        # Get walkable space next to dispenser, pot, ect
        valid_terrain = (' ', '1', '2')
        ny = len(self.mdp.terrain_mtx)-1
        nx = len(self.mdp.terrain_mtx[0])-1
        for d in Direction.ALL_DIRECTIONS:
            adj_goal_pos = Action.move_in_direction(asset_loc, d)
            if (0 <= adj_goal_pos[0] <= nx and 0 <= adj_goal_pos[1] <= ny):
                terrain = self.mdp.get_terrain_type_at_pos(adj_goal_pos)
                checked_terrains.append(terrain)
                if terrain in valid_terrain:
                    goal_pos = adj_goal_pos
                    break
        assert goal_pos is not None, f"Could not find valid position next to" \
                                     f" {goal}[{self.mdp.get_terrain_type_at_pos(asset_loc)}] at {asset_loc} \n" \
                                     f"\t checked: {checked_terrains}"

        # Find start positions within desired distance
        valid_start_pos = []
        for start_pos in self.mdp.get_valid_player_positions():
            hash_str = f'{goal_pos}_{start_pos}'
            dist = self.navigation_dist_hash.get(hash_str, self.mp.get_gridworld_pos_distance(start_pos, goal_pos))
            self.navigation_dist_hash[hash_str] = dist

            if dist < distance:
                valid_start_pos.append(start_pos)

        start_pos = valid_start_pos[np.random.randint(len(valid_start_pos))]

        # Reassign player if already in start pos
        for _ip, _player in enumerate(state.players):
            if _player.position == start_pos:
                self.assign_random_start_loc(_player, exclude=[start_pos])

        player.update_pos_and_or(start_pos, player.orientation)

        assert state.players[0].position != state.players[1].position, "Players cannot start at the same position"


    def add_onions_to_pots(self, state, n_onions,cooking_tick=None):
        assert n_onions >= 0 or n_onions <= 3, "Number of onions must be between 0 and 3"
        pots = self.mdp.get_pot_states(state)["empty"]
        # onion_quants = n_onions * np.eye(self.mdp.num_pots)[np.random.choice(self.mdp.num_pots)]  # which pot is filled
        onion_quants = n_onions * np.eye(len(pots),dtype=int)[np.random.choice(len(pots))]  # which pot is filled

        # assert len(onion_quants)==len(pots), "Number of pots must match number of onion quantities"
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
        """ Adds held objects to players in the state. If list of objects is 2D, randomly selects a posibility."""
        if len(np.shape(objs)) ==2:
            objs = objs[np.random.randint(len(objs))]
        elif len(np.shape(objs)) >2:
            raise ValueError(f'invalid dim {np.shape(objs)} for held object possibilities')

        # For each player, add a random object with prob rnd_obj_prob_thresh
        for obj,player in zip(objs,state.players):
            if obj is not None:
                self.add_held_obj(player, obj)
        return state

    def add_held_obj(self, player, obj):
        if obj == "soup":
            player.set_object(SoupState.get_soup(player.position, num_onions=3, num_tomatoes=0, finished=True))
        else:
            player.set_object(ObjectState(obj, player.position))
        return player

    def add_random_counter_state(self, state, rnd_obj_prob_thresh=0.025):
        """ adds random object to counters """
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

    #########################################################################################################
    # Curriculum Status  ###################################################################################
    def is_failing(self, training_iter, total_iter):
        """ checks if model learning is failing to progress efficiently"""
        training_prog = training_iter / total_iter
        if self.failure_thresh['enable']:
            return (training_prog > self.failure_thresh[self.curr_curriculum_name])
        else:
            return False

    def is_reset_needed(self, state, reward, info):
        """
        Determines when to reset game
        Goal of each curriculum is to deliver 1 soup except for full_task
        - slipping reset does not produce good results
        """

        if (self.reset_on_service
                and not self.sampled_curriculum == 'full_task'
                and self.sampling != 'uniform'
        ):
            was_soup_delivered = np.any(info['mdp_info']['event_infos']['soup_delivery'])
            if was_soup_delivered:  return True  # delivering soup
        return False

    ########################################################################################################
    # Properties ###########################################################################################
    @property
    def num_curriculums(self):
        return len(self.curriculums)

    @property
    def curr_curriculum_name(self):
        return self.curriculums[self.current_curriculum]

    @property
    def desired_dist2goal(self):
        it = self.subtask_iters[self.sampled_curriculum]
        prog = it / self.start_near_goal_iters
        # prog = self.iteration/self.start_near_goal_iters
        dist = np.clip(round(prog*self._max_dist2goal),1,self._max_dist2goal+1)
        return dist

    @property
    def is_latest_curriculum(self):
        if self.sampling == 'uniform': return True
        else: return self.sampled_curriculum == self.curr_curriculum_name

