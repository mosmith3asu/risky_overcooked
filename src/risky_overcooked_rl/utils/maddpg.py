import torch
import numpy as np
import torch.nn as nn
# from utils.misc import soft_update
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld,SoupState, ObjectState
from risky_overcooked_py.mdp.actions import Action
# from model.utils.model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torch import Tensor
import random
# from model.utils.model import fanin_init
import time
import numpy as np
import torch
import json, os
from risky_overcooked_rl.utils.rl_logger import RLLogger#,TrajectoryVisualizer, TrajectoryHeatmap
# from utils.train import make_dir
import logging

logger = logging.getLogger(__name__)
def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

class ReplayBuffer(object):

    def __init__(self, obs_shape, action_shape, reward_shape, dones_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        self.obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, *reward_shape), dtype=np.float32)
        self.dones = np.empty((capacity, *dones_shape), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, dones):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.dones[self.idx], dones)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size, nth=None):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)

        if nth:
            obses = torch.FloatTensor(self.obses[idxs][:, nth]).to(self.device)
            actions = torch.FloatTensor(self.actions[idxs][:, nth]).to(self.device)
            rewards = torch.FloatTensor(self.rewards[idxs][:, nth]).to(self.device)
            next_obses = torch.FloatTensor(self.next_obses[idxs][:, nth]).to(self.device)
            dones = torch.FloatTensor(self.dones[idxs][:, nth]).to(self.device)
        else:
            obses = torch.FloatTensor(self.obses[idxs]).to(self.device)
            actions = torch.FloatTensor(self.actions[idxs]).to(self.device)
            rewards = torch.FloatTensor(self.rewards[idxs]).to(self.device)
            next_obses = torch.FloatTensor(self.next_obses[idxs]).to(self.device)
            dones = torch.FloatTensor(self.dones[idxs]).to(self.device)

        return obses, actions, rewards, next_obses, dones

    def save(self, root_dir, step) -> None:
        make_dir(root_dir, 'buffer') if root_dir else None
        length = self.capacity if self.full else self.idx
        path = os.path.join(root_dir, 'buffer')

        make_dir(path, str(step))
        path = os.path.join(path, str(step))

        np.savez_compressed(os.path.join(path, 'state.npz'), self.obses)
        np.savez_compressed(os.path.join(path, 'next_state.npz'), self.next_obses)
        np.savez_compressed(os.path.join(path, 'action.npz'), self.actions)
        np.savez_compressed(os.path.join(path, 'reward.npz'), self.rewards)
        np.savez_compressed(os.path.join(path, 'done.npz'), self.dones)

        info = dict()
        info['idx'] = self.idx
        info['capacity'] = self.capacity
        info['step'] = step
        info['size'] = length

        with open(os.path.join(path, 'info.txt'), 'w') as f:
            output = json.dumps(info, indent=4, sort_keys=True)
            f.write(output)

    def load(self, root_dir) -> None:
        path = os.path.join(root_dir, 'buffer')

        self.obses = np.load(os.path.join(path, 'state.npz'))['arr_0']
        self.next_obses = np.load(os.path.join(path, 'next_state.npz'))['arr_0']
        self.actions = np.load(os.path.join(path, 'action.npz'))['arr_0']
        self.rewards = np.load(os.path.join(path, 'reward.npz'))['arr_0']
        self.dones = np.load(os.path.join(path, 'done.npz'))['arr_0']

        with open(os.path.join(path, 'info.txt'), 'r') as f:
            info = json.load(f)

        self.idx = int(info['idx'])
        self.capacity = int(info['capacity'])
        self.full = int(info['step']) >= self.capacity

    def append_data(self, dir_path):

        def loader(path):
            logger.info('Loading data - ' + path)
            data =  np.load(path)['arr_0']
            logger.info('Loaded data - ' + path)
            return data

        obses_data = loader(os.path.join(dir_path, 'state.npz'))
        self.obses = np.concatenate((self.obses, obses_data), axis=0)

        next_obses_data = loader(os.path.join(dir_path, 'next_state.npz'))
        self.next_obses = np.concatenate((self.next_obses, next_obses_data), axis=0)

        reward_data = loader(os.path.join(dir_path, 'reward.npz'))
        self.rewards = np.concatenate((self.rewards, reward_data), axis=0)

        action_data = loader(os.path.join(dir_path, 'action.npz'))
        self.actions = np.concatenate((self.actions, action_data), axis=0)

        done_data = loader(os.path.join(dir_path, 'done.npz'))
        self.dones = np.concatenate((self.dones, done_data), axis=0)

        if self.idx == 0:
            self.idx = -1
        self.idx += len(obses_data)
# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
def number_to_onehot(X):
    is_batch = X.dim() == 3
    is_torch = type(X) == torch.Tensor

    if is_batch:
        num_agents = X.shape[1]
        X = X.reshape(-1, 1)

    shape = (X.shape[0], int(X.max() + 1))
    one_hot = np.zeros(shape)
    rows = np.arange(X.shape[0])

    positions = X.reshape(-1).cpu().detach().numpy().astype(int)
    one_hot[rows, positions] = 1.

    if is_batch:
        one_hot = one_hot.reshape(-1, num_agents, int(X.max() + 1))

    if is_torch:
        one_hot = torch.Tensor(one_hot).to(X.device)

    return one_hot
def onehot_to_number(X):
    if type(X) == torch.Tensor:
        return torch.argmax(X, dim=1)
    else:
        return np.argmax(X, axis=1)
class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
    return F.softmax(y / temperature, dim=1)
# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=256, activation=F.relu,
                 constrain_out=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        self.norm1 = nn.BatchNorm1d(input_dim)
        self.norm1.weight.data.fill_(1)
        self.norm1.bias.data.fill_(0)

        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, out_dim)

        self.activation = activation

        # self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        # self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.fc4.weight.data = fanin_init(self.fc4.weight.data.size())

        if constrain_out:
            # self.fc3.weight.data.uniform_(-0.003, 0.003)
            self.fc5.weight.data.uniform_(-0.003, 0.003)
            self.out_fn = torch.tanh
        else:
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        # X = self.norm1(X)
        h1 = self.activation(self.fc1(X))
        h2 = self.activation(self.fc2(h1))
        h3 = self.activation(self.fc3(h2))
        h4 = self.activation(self.fc4(h3))
        out = self.out_fn(self.fc5(h4))
        return out


class DDPGAgent(nn.Module):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """

    def __init__(self, params):
        super(DDPGAgent, self).__init__()
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """

        self.lr = params.lr
        self.gamma = params.gamma

        self.obs_dim = params.obs_dim
        self.action_dim = params.action_dim
        self.device = params.device
        self.discrete_action = params.discrete_action_space
        self.hidden_dim = params.hidden_dim

        constrain_out = not self.discrete_action

        self.policy = MLPNetwork(self.obs_dim, self.action_dim,
                                 hidden_dim=self.hidden_dim,
                                 constrain_out=constrain_out)
        self.critic = MLPNetwork(params.critic.obs_dim, 1,
                                 hidden_dim=self.hidden_dim,
                                 constrain_out=False)
        self.target_policy = MLPNetwork(self.obs_dim, self.action_dim,
                                        hidden_dim=self.hidden_dim,
                                        constrain_out=constrain_out)
        self.target_critic = MLPNetwork(params.critic.obs_dim, 1,
                                        hidden_dim=self.hidden_dim,
                                        constrain_out=False)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.lr * 0.1)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)

        self.exploration = OUNoise(self.action_dim)

        self.num_heads = 100

    def act(self, obs, explore=False):

        if obs.dim() == 1:
            obs = obs.unsqueeze(dim=0)

        action = self.policy(obs)

        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)

            action = onehot_to_number(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()), requires_grad=False).to(action.device)
            action = action.clamp(-1, 1)

        return action.detach().cpu().numpy()

    def reset_noise(self):
        self.exploration.reset()

    def scale_noise(self, scale):
        self.exploration.scale = scale

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

class MADDPG(object):

    def __init__(self, name, params):

        self.name = name
        self.lr = params.lr
        self.gamma = params.gamma
        self.tau = params.tau

        self.obs_dim = params.obs_dim
        self.action_dim = params.action_dim
        self.batch_size = params.batch_size
        self.device = params.device
        self.discrete_action = params.discrete_action_space

        self.agent_index = params.agent_index
        self.num_agents = len(self.agent_index)

        self.mse_loss = torch.nn.MSELoss()

        # Reshape critic input shape for shared observation
        params.critic.obs_dim = (self.obs_dim + self.action_dim) * self.num_agents

        self.agents = [DDPGAgent(params) for _ in range(self.num_agents)]
        [agent.to(self.device) for agent in self.agents]

    def scale_noise(self, scale):
        for agent in self.agents:
            agent.scale_noise(scale)

    def reset_noise(self):
        for agent in self.agents:
            agent.reset_noise()

    def act(self, observations, sample=False):
        observations = torch.Tensor(observations).to(self.device)

        actions = []
        for agent, obs in zip(self.agents, observations):
            agent.eval()
            actions.append(agent.act(obs, explore=sample).squeeze())
            agent.train()
        return np.array(actions)

    def update(self, replay_buffer, logger, step):

        sample = replay_buffer.sample(self.batch_size, nth=self.agent_index)
        obses, actions, rewards, next_obses, dones = sample

        if self.discrete_action:  actions = number_to_onehot(actions)

        for agent_i, agent in enumerate(self.agents):

            ''' Update value '''
            agent.critic_optimizer.zero_grad()

            with torch.no_grad(): #TODO: remove redundant torch->numpy
                if self.discrete_action:
                    # Grab ordered target actions for agent i -----
                    target_actions = torch.Tensor(
                        np.array([onehot_from_logits(policy(next_obs)).detach().cpu().numpy() for policy, next_obs in
                                               zip(self.target_policies, torch.swapaxes(next_obses, 0, 1))])
                    ).to(self.device)
                else:
                    target_actions = torch.Tensor(
                        np.array([policy(next_obs).detach().cpu().numpy() for policy, next_obs in
                                                   zip(self.target_policies, torch.swapaxes(next_obses, 0, 1))])
                    ).to(self.device)
                target_actions = torch.swapaxes(target_actions, 0, 1)
                target_critic_in = torch.cat((next_obses, target_actions), dim=2).view(self.batch_size, -1)

                ############ INSERT CPT ####################
                target_next_q = rewards[:, agent_i] + (1 - dones[:, agent_i]) * self.gamma * agent.target_critic(target_critic_in)

            critic_in = torch.cat((obses, actions), dim=2).view(self.batch_size, -1)
            main_q = agent.critic(critic_in)

            critic_loss = self.mse_loss(main_q, target_next_q)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
            agent.critic_optimizer.step()

            ''' Update policy '''
            agent.policy_optimizer.zero_grad()

            policy_out = agent.policy(obses[:, agent_i])
            if self.discrete_action:
                action = gumbel_softmax(policy_out, hard=True)
            else:
                action = policy_out

            joint_actions = torch.zeros((self.batch_size, self.num_agents, self.action_dim)).to(self.device)
            for i, policy, local_obs, act in zip(range(self.num_agents), self.policies, torch.swapaxes(obses, 0, 1), torch.swapaxes(actions, 0, 1)):
                if i == agent_i:
                    joint_actions[:, i] = action
                else:
                    other_action = onehot_from_logits(policy(local_obs)) if self.discrete_action else policy(local_obs)
                    joint_actions[:, i] = other_action

            critic_in = torch.cat((obses, joint_actions), dim=2).view(self.batch_size, -1)

            actor_loss = -agent.critic(critic_in).mean()
            actor_loss += (policy_out ** 2).mean() * 1e-3  # Action regularize
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 0.5)
            agent.policy_optimizer.step()

        self.update_all_targets()

    def update_all_targets(self):
        for agent in self.agents:
            soft_update(agent.target_critic, agent.critic, self.tau)
            soft_update(agent.target_policy, agent.policy, self.tau)

    def save(self, step):
        # os.mk
        #
        # for i, agent in self.agents:
        #     name = '{0}_{1}_{step}.pth'.format(self.name, i, step)
        #     torch.save(agent, )
        #
        #
        # raise NotImplementedError
        pass

    def load(self, filename):

        raise NotImplementedError

    @property
    def policies(self):
        return [agent.policy for agent in self.agents]

    @property
    def target_policies(self):
        return [agent.target_policy for agent in self.agents]

    @property
    def critics(self):
        return [agent.critic for agent in self.agents]

    @property
    def target_critics(self):
        return [agent.target_critic for agent in self.agents]
import itertools
class Trainer():
    def __init__(self,cfg):
        self.cfg = cfg
        self.n_agents = 2
        # logger --
        set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.discrete_action = cfg.discrete_action_space

        # Create env
        self.mdp = OvercookedGridworld.from_layout_name(cfg.layout)
        self.mdp.p_slip = cfg.p_slip
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=cfg.episode_length, time_cost=0)

        # Define agents
        self.agent_indexes = [0,1]
        self.adversary_indexes = []


        # OU Noise settings
        self.num_seed_steps = 10000#cfg.num_seed_steps
        self.ou_exploration_steps = cfg.num_train_steps
        self.ou_init_scale = 0.3#cfg.ou_init_scale
        self.ou_final_scale = 0#cfg.ou_final_scale

        self.explore_rate = 0.0

        # if self.discrete_action:
        #     cfg.agent.params.obs_dim = self.mdp.get_lossless_encoding_vector_shape()
        #     cfg.agent.params.action_dim = Action.NUM_ACTIONS
        #     cfg.agent.params.action_range =  list(range(cfg.agent.params.action_dim))
        # else:
        #     cfg.agent.params.obs_dim = self.env.observation_space[0].shape[0]
        #     cfg.agent.params.action_dim = self.env.action_space[0].shape[0]
        #     cfg.agent.params.action_range = [-1, 1]

        cfg.agent.params.obs_dim = self.mdp.get_lossless_encoding_vector_shape()[0]
        cfg.agent.params.action_dim = Action.NUM_ACTIONS
        cfg.agent.params.action_range = list(range(cfg.agent.params.action_dim))
        self.joint_action_space = list(itertools.product(Action.ALL_ACTIONS, repeat=2))

        cfg.agent.params.agent_index = self.agent_indexes
        cfg.agent.params.critic.input_dim = cfg.agent.params.obs_dim + cfg.agent.params.action_dim

        # self.agent = hydra.utils.instantiate(cfg.agent)

        self.agent = MADDPG(cfg.agent.name, cfg.agent.params)
        self.common_reward = cfg.common_reward
        obs_shape = [self.n_agents, cfg.agent.params.obs_dim]
        action_shape = [self.n_agents, cfg.agent.params.action_dim if not self.discrete_action else 1]
        reward_shape = [self.n_agents, 1]
        dones_shape = [self.n_agents, 1]
        self.replay_buffer = ReplayBuffer(obs_shape=obs_shape,
                                          action_shape=action_shape,
                                          reward_shape=reward_shape,
                                          dones_shape=dones_shape,
                                          capacity=int(cfg.replay_buffer_capacity),
                                          device=self.device)

        # for dir_path in cfg.data.data_dirs:
        #     self.replay_buffer.append_data(dir_path)

        self.logger = None
        self.video_recorder = None#VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0
        self.estimated_step = 0

        self.logger = RLLogger(rows=3, cols=1, num_iterations=self.cfg.num_train_steps)
        self.logger.add_lineplot('test_reward', xlabel='', ylabel='$R_{test}$', filter_window=30, display_raw=True,
                                 loc=(0, 1))
        self.logger.add_lineplot('train_reward', xlabel='', ylabel='$R_{train}$', filter_window=30, display_raw=True,
                                 loc=(1, 1))
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
        for episode in range(self.cfg.num_eval_episodes):
            self.env.reset()
            obs = self.mdp.get_lossless_encoding_vector(self.env.state)
            obs = np.vstack([obs, self.invert_obs(obs)])
            episode_step = 0

            done = False
            episode_reward = 0
            while not done:
                obs = self.mdp.get_lossless_encoding_vector(self.env.state)
                obs = np.vstack([obs, self.invert_obs(obs)])
                action = self.agent.act(obs, sample=False)
                joint_action_idx = Action.INDEX_TO_ACTION_INDEX_PAIRS.index(tuple(action))
                joint_action = self.joint_action_space[joint_action_idx]
                next_state, rewards, done, info = self.env.step(joint_action)
                # next_state, rewards, done, info = self.env.step(action)

                # # rewards = np.array(info['shaped_r_by_agent']).reshape(-1, 1)
                # episode_reward += sum(rewards)[0]
                episode_reward += np.mean(rewards)
                episode_step += 1

            average_episode_reward += episode_reward
        # self.video_recorder.save(f'{self.step}.mp4')

        average_episode_reward /= self.cfg.num_eval_episodes
        print(f'eval/episode_reward, {average_episode_reward}, {self.step}\n')
        # self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        # self.logger.dump(self.step)
        return average_episode_reward

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        train_reward_buffer = []
        while self.step < self.cfg.num_train_steps + 1:
            self.logger.spin()
            train_reward_buffer.append(episode_reward)



            if done or self.step % self.cfg.eval_frequency == 0:

                print('train/episode_reward', episode_reward, self.step)
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
                            self.step - self.num_seed_steps)) / self.ou_exploration_steps
                self.agent.scale_noise(
                    self.ou_final_scale + (self.ou_init_scale - self.ou_final_scale) * self.ou_percentage)
                self.agent.reset_noise()

                episode_reward = 0
                episode_step = 0
                episode += 1

                # self.logger.log('train/episode', episode, self.step)
                # print('train/episode_reward', episode_reward, self.step)
            # Warmup ----------------
            if self.step < self.num_seed_steps or self.explore_rate > np.random.rand():
                # action = np.array([self.env.action_space.sample() for _ in self.env_agent_types])
                action = np.array([np.random.choice(np.arange(len(Action.ALL_ACTIONS))) for _ in range(self.n_agents)])
                if self.discrete_action: action = action.reshape(-1, 1)
            # Act
            else:
                agent_observation = obs[self.agent_indexes]
                agent_actions = self.agent.act(agent_observation, sample=True)
                action = agent_actions

            if self.step >= self.num_seed_steps and self.step >= self.agent.batch_size:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            joint_action_idx = Action.INDEX_TO_ACTION_INDEX_PAIRS.index(tuple(action))
            joint_action = self.joint_action_space[joint_action_idx]
            next_state, rewards, done, info = self.env.step(joint_action)
            next_obs = self.mdp.get_lossless_encoding_vector(self.env.state)
            next_obs = np.vstack([next_obs, self.invert_obs(next_obs)])
            rewards = np.array(info['shaped_r_by_agent']).reshape(-1, 1)

            if episode_step + 1 == self.env.horizon:#== self.env.episode_length:
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

    def invert_obs(self, obs):
        N_PLAYER_FEAT = 9
        _obs = np.concatenate([obs[N_PLAYER_FEAT:2 * N_PLAYER_FEAT],
                               obs[:N_PLAYER_FEAT],
                               obs[2 * N_PLAYER_FEAT:]])
        # _obs = np.cat([obs[:, N_PLAYER_FEAT:2 * N_PLAYER_FEAT],
        #                        obs[:, :N_PLAYER_FEAT],
        #                        obs[:, 2 * N_PLAYER_FEAT:]], dim=1)
        return _obs

class Config():
    class agent:
        name = 'maddpg'
        _target_ = MADDPG

        class params:
            obs_dim = None
            action_dim= None  # to be specified later
            action_range= None  # to be specified later
            agent_index= None  # Different by environments
            hidden_dim= 256
            device= 'cuda'
            discrete_action_space= True
            batch_size= 256
            lr= 0.001
            tau=0.01
            gamma= 0.95
            class critic:
                input_dim = None




    data = 'local'
    layout = 'risky_coordination_ring'
    p_slip = 0.0

    discrete_action = True
    discrete_action_space = True
    episode_length= 400
    experiment= 'vanilla'
    seed= 0

    num_train_steps= 500_000#40000

    eval_frequency= 5000
    num_eval_episodes= 3

    common_reward= False#True

    device = "cuda"

    # Logging Settings
    log_frequency= 1000
    log_save_tb= False
    save_video= False

    replay_buffer_capacity =5e4



def main() -> None:
    cfg = Config()
    trainer = Trainer(cfg)
    trainer.run()



if __name__=="__main__":
    main()