import itertools
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld,SoupState, ObjectState
from risky_overcooked_py.mdp.actions import Action
import time
from risky_overcooked_rl.utils.rl_logger import RLLogger

from collections import deque
from src.risky_overcooked_rl.algorithms.MADDPG.utils import *
from src.risky_overcooked_rl.algorithms.MADDPG.memory import ReplayMemory#ReplayBuffer
from src.risky_overcooked_rl.algorithms.MADDPG.agents import MADDPG

class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_agents = 2
        # logger --
        set_seed_everywhere(cfg.env.seed)
        self.device = torch.device(cfg.device)
        cfg.agent.device = self.device
        # self.discrete_action = cfg.discrete_action_space

        # Create env
        self.mdp = OvercookedGridworld.from_layout_name(cfg.env.layout)
        self.mdp.p_slip = cfg.env.p_slip
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=cfg.env.horizon, time_cost=0)

        # Define agents
        # self.agent_indexes = [0, 1]
        # self.adversary_indexes = []

        # OU Noise settings
        self.num_episodes = cfg.env.num_episodes
        self.num_eval_episodes = cfg.env.num_eval_episodes
        self.eval_episode_freq = cfg.env.eval_episode_freq
        self.num_warmup_episodes = cfg.env.num_warmup_episodes
        self.num_warmup_steps = cfg.env.num_warmup_episodes * self.env.horizon

        self.noise = cfg.env.exploration_noise
        self.rshape = cfg.env.reward_shaping
        # self.ou_init_scale = cfg.noise.init_scale
        # self.ou_final_scale = cfg.noise.final_scale
        # self.ou_exp_decay = cfg.noise.exponent_decay

        cfg.agent.obs_dim = self.mdp.get_lossless_encoding_vector_shape()[0]
        cfg.agent.action_dim = Action.NUM_ACTIONS
        cfg.agent.action_range = list(range(cfg.agent.action_dim))
        self.joint_action_space = list(itertools.product(Action.ALL_ACTIONS, repeat=2))

        # cfg.agent.params.agent_index = self.agent_indexes
        cfg.agent.critic.input_dim = cfg.agent.obs_dim + 2*cfg.agent.action_dim


        self.agent = MADDPG(cfg.agent.name, cfg.agent)
        self.replay_buffer = ReplayMemory(int(float(cfg.env.replay_buffer_capacity)),self.device)


        # self.step = 0
        # self.estimated_step = 0

        self.logger = RLLogger(rows=3, cols=1, num_iterations=self.num_episodes)
        self.logger.add_lineplot('test_reward', xlabel='', ylabel='$R_{test}$', filter_window=30, display_raw=True, loc=(0, 1))
        self.logger.add_lineplot('train_reward', xlabel='', ylabel='$R_{train}$', filter_window=30, display_raw=True,  loc=(1, 1))
        # self.logger.add_lineplot('loss', xlabel='iter', ylabel='$Loss$', filter_window=30, display_raw=True, loc=(2, 1))
        # self.logger.add_checkpoint_line()
        # self.logger.add_table('Params', config)
        self.logger.add_status()
        # self.logger.add_button('Preview', callback=self.traj_visualizer.preview_qued_trajectory)
        # self.logger.add_button('Heatmap', callback=self.traj_heatmap.preview)
        # self.logger.add_button('Save ', callback=self.save)

    def evaluate(self):
        average_episode_reward = 0

        # self.video_recorder.init(enabled=True)
        for episode in range(self.num_eval_episodes):
            self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                obs = self.mdp.get_lossless_encoding_vector_astensor(self.env.state, self.device)
                agent_observations = torch.vstack([obs, invert_obs(obs)])
                action = self.agent.act(agent_observations, sample=False)
                joint_action_idx = Action.INDEX_TO_ACTION_INDEX_PAIRS.index(tuple(action))
                joint_action = self.joint_action_space[joint_action_idx]
                next_state, rewards, done, info = self.env.step(joint_action)
                episode_reward += np.mean(rewards)
            average_episode_reward += episode_reward
        average_episode_reward /= self.num_eval_episodes
        print(f'\neval/episode_reward, {average_episode_reward}\n')
        return average_episode_reward

    def warmup(self):
        # train_sparse_rewards = deque(maxlen=self.env.horizon)
        # train_shaped_rewards = deque(maxlen=self.env.horizon)
        # train_loss = deque(maxlen=self.env.horizon)

        for _ in range(self.num_warmup_episodes):
            episode_sparse_reward = 0
            episode_shaped_reward = 0
            self.env.reset()
            # obs = self.mdp.get_lossless_encoding_vector(self.env.state)
            obs = self.mdp.get_lossless_encoding_vector_astensor(self.env.state, device=self.device)

            for t in range(self.env.horizon):
                # Act
                action = np.array([np.random.choice(np.arange(len(Action.ALL_ACTIONS))) for _ in range(self.n_agents)])
                joint_action_idx = Action.INDEX_TO_ACTION_INDEX_PAIRS.index(tuple(action))
                joint_action = self.joint_action_space[joint_action_idx]

                # Step state-action
                next_state, sparse_rewards, done, info = self.env.step(joint_action)
                # next_obs = self.mdp.get_lossless_encoding_vector(self.env.state)
                next_obs = self.mdp.get_lossless_encoding_vector_astensor(self.env.state, device=self.device)
                shaped_rewards = info['shaped_r_by_agent']

                # Log and store in memory and update
                episode_sparse_reward += sparse_rewards
                episode_shaped_reward += np.mean(shaped_rewards)
                total_rewards = np.mean(sparse_rewards) + np.array(shaped_rewards).reshape(-1, 1)
                self.replay_buffer.double_push(obs, joint_action_idx,total_rewards, next_obs,done)

                # Close step
                obs = next_obs
                # self.step += 1
                if done:
                    print('warmup/episode_reward', [episode_sparse_reward, episode_shaped_reward])
                    break

    def run(self):
        sparse_rewards_buffer = []
        shaped_rewards_buffer = []
        # loss_buffer = []

        self.warmup()
        for epi in range(self.num_episodes):
            self.logger.start_iteration()

            # Exploration noise
            # percentage = max(0, self.ou_exploration_steps - (self.step - self.num_warmup_steps)) / self.ou_exploration_steps
            prog = epi/self.num_episodes
            self.noise_percentage = prog ** self.noise.exp_decay
            self.agent.scale_noise( self.noise.final_scale + (self.noise.init_scale -  self.noise.final_scale) * self.noise_percentage)
            self.agent.reset_noise()

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
                joint_action  = self.joint_action_space[joint_action_idx]

                # Step state-action
                next_state, sparse_rewards, done, info = self.env.step(joint_action)
                next_obs = self.mdp.get_lossless_encoding_vector_astensor(self.env.state,self.device)
                shaped_rewards = info['shaped_r_by_agent']

                # Log and store in memory and update
                episode_sparse_reward += sparse_rewards
                episode_shaped_reward += np.mean(shaped_rewards)
                total_rewards = np.mean(sparse_rewards) + np.array(shaped_rewards).reshape(-1, 1)
                self.replay_buffer.double_push(obs, joint_action_idx,total_rewards, next_obs,done)
                self.agent.update(self.replay_buffer)

                # Close step
                obs = next_obs
                if done:
                    print(f'[{round(prog*100,1)}% | epi: {epi}] train/episode_reward', [episode_sparse_reward, episode_shaped_reward])
                    break

            # Log episode stats
            sparse_rewards_buffer.append(episode_sparse_reward)
            shaped_rewards_buffer.append(episode_shaped_reward)

            # EVALUATION ROLLOUT ##################################

            if epi % self.eval_episode_freq == 0:
                ave_eval_reward = self.evaluate()
                self.logger.log(
                    test_reward=[epi, np.mean(ave_eval_reward)],
                    train_reward=[epi, np.mean(sparse_rewards_buffer) + np.mean(shaped_rewards_buffer)],
                )
                self.logger.draw()
                sparse_rewards_buffer = []
                shaped_rewards_buffer = []
            self.logger.end_iteration()

        self.logger.wait_for_close(enable=True)

    def run_OG(self):
        episode, episode_reward, done = 0, 0, True
        sparse_episode_reward = 0
        start_time = time.time()
        train_reward_buffer = []
        self.logger.start_iteration()
        while self.step < self.cfg.num_train_steps + 1:
            self.logger.spin()

            train_reward_buffer.append(episode_reward)
            if done:
                self.logger.end_iteration()
                self.logger.start_iteration()

            if done or self.step % self.cfg.eval_frequency == 0:

                print('train/episode_reward', [sparse_episode_reward, episode_reward - sparse_episode_reward],
                      self.step)
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    ave_eval_reward = self.evaluate()

                    self.logger.log(
                        test_reward=[self.step, np.mean(ave_eval_reward)],
                        train_reward=[self.step, np.mean(train_reward_buffer)],
                    )
                    self.logger.draw()
                    train_reward_buffer = []

                self.env.reset()
                obs = self.mdp.get_lossless_encoding_vector(self.env.state)
                obs = np.vstack([obs, self.invert_obs(obs)])

                self.ou_percentage = max(0, self.ou_exploration_steps - (
                            self.step - self.num_warmup_steps)) / self.ou_exploration_steps
                self.ou_percentage = self.ou_percentage ** self.ou_exp_decay
                self.agent.scale_noise(
                    self.ou_final_scale + (self.ou_init_scale - self.ou_final_scale) * self.ou_percentage)
                self.agent.reset_noise()

                reward_scale = self.ou_percentage

                sparse_episode_reward = 0
                episode_reward = 0
                episode_step = 0
                episode += 1

                # self.logger.log('train/episode', episode, self.step)
                # print('train/episode_reward', episode_reward, self.step)
            # Warmup ----------------
            if self.step < self.num_warmup_steps:
                # action = np.array([self.env.action_space.sample() for _ in self.env_agent_types])
                action = np.array([np.random.choice(np.arange(len(Action.ALL_ACTIONS))) for _ in range(self.n_agents)])
                if self.discrete_action: action = action.reshape(-1, 1)
            # Act
            else:
                agent_observation = obs[self.agent_indexes]
                agent_actions = self.agent.act(agent_observation, sample=True)
                action = agent_actions

            if self.step >= self.num_warmup_steps and self.step >= self.agent.batch_size:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            joint_action_idx = Action.INDEX_TO_ACTION_INDEX_PAIRS.index(tuple(action))
            joint_action = self.joint_action_space[joint_action_idx]
            next_state, rewards, done, info = self.env.step(joint_action)
            next_obs = self.mdp.get_lossless_encoding_vector(self.env.state)
            next_obs = np.vstack([next_obs, self.invert_obs(next_obs)])

            sparse_episode_reward += rewards
            rewards += np.array(info['shaped_r_by_agent']).reshape(-1, 1)  # *reward_scale

            if episode_step + 1 == self.env.horizon:  # == self.env.episode_length:
                done = True

            # if self.cfg.render:
            #     cv2.imshow('Overcooked', self.env.render())
            #     cv2.waitKey(1)

            # episode_reward += sum(rewards)[0]
            episode_reward += np.mean(rewards)

            if self.discrete_action: action = action.reshape(-1, 1)

            dones = np.array([done for _ in range(self.n_agents)]).reshape(-1, 1)

            self.replay_buffer.add(obs, action, rewards, next_obs, dones)

            obs = next_obs
            episode_step += 1
            self.step += 1

            # if self.step % 5e4 == 0 and self.save_replay_buffer:
            #     self.replay_buffer.save(self.work_dir, self.step - 1)
        self.logger.wait_for_close(enable=True)



def main():
    pass


def subfun():
    pass


if __name__ == "__main__":
    main()
