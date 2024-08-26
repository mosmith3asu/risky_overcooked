import numpy as np
from risky_overcooked_py.mdp.overcooked_mdp import Recipe,OvercookedGridworld
from risky_overcooked_rl.utils.trainer import Trainer
import numpy as np
from risky_overcooked_rl.utils.deep_models import device,SelfPlay_QRE_OSA,SelfPlay_QRE_OSA_CPT

from datetime import datetime
debug = False
from collections import deque
class CirriculumTrainer(Trainer):
    def __init__(self,model_object,config):
        super().__init__(model_object,config)
        reward_thresh = config['reward_thresh'] if 'reward_thresh' in config else 40
        self.curriculum = Curriculum(self.env,reward_thresh,config)
        self.schedule_decay = 0.7


    def run(self):
        train_rewards = []
        train_losses = []
        # Main training Loop
        for it in range(self.ITERATIONS):
            cit = self.curriculum.iteration

            self.logger.spin()

            # Training Step ##########################################
            # Set Iteration parameters
            self._epsilon = self.epsilon_sched[cit]
            self._rationality = self.rationality_sched[cit]
            self._rshape_scale = self.rshape_sched[cit]
            self._p_rand_start = self.random_start_sched[cit]

            # Perform Rollout
            self.logger.start_iteration()
            cum_reward, cum_shaped_rewards, rollout_info = self.training_rollout(cit)
            if it > 1: self.model.scheduler.step()  # updates learning rate scheduler
            self.model.update_target()  # performs soft update of target network
            self.logger.end_iteration()

            next_cirriculum = self.curriculum.step_cirriculum(cum_reward)
            if next_cirriculum:
                self.init_sched(self.config, eps_decay=self.schedule_decay, rshape_decay=self.schedule_decay)
            slips = rollout_info['onion_slip'] + rollout_info['dish_slip'] + rollout_info['soup_slip']
            risks = rollout_info['onion_risked'] + rollout_info['dish_risked'] + rollout_info['soup_risked']
            handoffs = rollout_info['onion_handoff'] + rollout_info['dish_handoff'] + rollout_info['soup_handoff']

            print(f"[it:{it}"
                  f" cur:{self.curriculum.current_cirriculum}-{cit}]"
                  f"[R:{round(cum_reward, 3)} "
                  f" Rshape:{np.round(cum_shaped_rewards, 3)} "
                  f" L:{round(rollout_info['mean_loss'], 3)} ]"
                  # f"| slips:{slips} "
                  f"[ risks:{risks} "
                  f" handoffs:{handoffs} ]"
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
                self.curriculum.eval('on')

                # Rollout test episodes ----------------------
                test_rewards = []
                test_shaped_rewards = []
                self._rationality = self.test_rationality
                self._epsilon = 0
                self._rshape_scale = 1
                for test in range(self.N_tests):
                    test_reward, test_shaped_reward, state_history, action_history, aprob_history = self.test_rollout()
                    test_rewards.append(test_reward)
                    test_shaped_rewards.append(test_shaped_reward)
                    if not self.has_checkpointed:
                        self.traj_visualizer.que_trajectory(state_history)

                # Checkpointing ----------------------
                self.test_rewards.append(np.mean(test_rewards))  # for checkpointing
                self.train_rewards.append(np.mean(train_rewards))  # for checkpointing
                if self.checkpoint(it):  # check if should checkpoint
                    self.traj_visualizer.que_trajectory(state_history)  # load preview of checkpointed trajectory

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
                self.curriculum.eval('off')
        self.logger.wait_for_close(enable=True)
class Curriculum:
    def __init__(self, env, reward_thresh=40, config=None):
        self.env = env
        self.mdp = env.mdp
        self.layout = self.mdp.layout_name
        self.reward_thresh = reward_thresh
        self.reward_buffer = deque(maxlen=10)
        self.iteration = 0 #iteration for this cirriculum
        self.min_iterations_per_cirriculum = 100
        self.default_params = {'n_onions': 3, 'cook_time': 20}

        self.cirriculums = [
            {'n_onions': 1, 'cook_time': 1},
            {'n_onions': 2, 'cook_time': 1},
            {'n_onions': 3, 'cook_time': 1},
            {'n_onions': 3, 'cook_time': 20},
        ]
        self.current_cirriculum = 0
        self.set_cirriculum(**self.cirriculums[self.current_cirriculum]) # set initial cirriculum

    def step_cirriculum(self, reward):
        self.reward_buffer.append(reward)
        if (np.mean(self.reward_buffer) >= self.reward_thresh
                and self.current_cirriculum < len(self.cirriculums)
                and self.iteration > self.min_iterations_per_cirriculum
        ):
            self.set_cirriculum(**self.cirriculums[self.current_cirriculum])
            self.current_cirriculum += 1
            self.iteration = 0
            return True
        self.iteration += 1
        return False
    def set_cirriculum(self, n_onions, cook_time):
        Recipe.MAX_NUM_INGREDIENTS = n_onions
        recipe_config = {
            'start_all_orders': [{'ingredients': ['onion' for _ in range(n_onions)]}],
            'num_items_for_soup': n_onions,
            "recipe_times": [cook_time],
        }
        self.mdp._configure_recipes(**recipe_config)
        self.mdp.num_items_for_soup = n_onions
        self.mdp.start_all_orders = recipe_config['start_all_orders']

    def eval(self,status):
        if status.lower() == 'on': self.set_cirriculum(**self.default_params)
        elif status.lower() == 'off':  self.set_cirriculum(**self.cirriculums[self.current_cirriculum])
        else: raise ValueError(f"Invalid curriculum test mode status '{status}'. Use 'on' or 'off'")
def main():
    config = {
        'ALGORITHM': 'Boltzmann_QRE-DDQN-OSA',
        'Date': datetime.now().strftime("%m_%d_%Y-%H_%M"),

        # Env Params ----------------
        # 'LAYOUT': "risky_coordination_ring",
        # 'LAYOUT': "risky_multipath",
        'LAYOUT': "forced_coordination",
        # 'LAYOUT': "forced_coordination_sanity_check",
        'HORIZON': 200,
        'ITERATIONS': 30_000,
        'AGENT': None,  # name of agent object (computed dynamically)
        "obs_shape": None,  # computed dynamically based on layout
        "p_slip": 0.1,

        # Learning Params ----------------
        "rand_start_sched": [0.1, 0.1, 10_000],  # percentage of ITERATIONS with random start states
        'epsilon_sched': [0.9, 0.15, 5000],  # epsilon-greedy range (start,end)
        'rshape_sched': [1, 0, 10_000],  # rationality level range (start,end)
        'rationality_sched': [5, 5, 10_000],
        'lr_sched': [1e-4, 1e-4, 1],
        'gamma': 0.97,  # discount factor
        'tau': 0.01,  # soft update weight of target network
        "num_hidden_layers": 5,  # MLP params
        "size_hidden_layers": 256,  # 32,      # MLP params
        "device": device,
        "minibatch_size": 256,  # size of mini-batches
        "replay_memory_size": 20_000,  # size of replay memory
        'clip_grad': 100,
        'monte_carlo': False,
        'note': 'Cirriculum OSA',
    }

    # config['LAYOUT'] = 'forced_coordination'; config['rand_start_sched'] = [0,0,1]
    config['LAYOUT'] = 'risky_coordination_ring'; config['tau'] = 0.005
    # config['LAYOUT'] = 'risky_multipath'
    # config['LAYOUT'] = 'forced_coordination_sanity_check'; config['rand_start_sched'] = [0,0,1]
    CirriculumTrainer(SelfPlay_QRE_OSA, config).run()

if __name__ == '__main__':
    main()
