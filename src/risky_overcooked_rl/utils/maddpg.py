import torch
import numpy as np
import torch.nn as nn
from utils.misc import soft_update


from model.utils.model import *
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils.model import fanin_init


class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, activation=F.relu,
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

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

        self.activation = activation

        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        if constrain_out:
            self.fc3.weight.data.uniform_(-0.003, 0.003)
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
        out = self.out_fn(self.fc3(h2))
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

            with torch.no_grad():
                if self.discrete_action:
                    target_actions = torch.Tensor([onehot_from_logits(policy(next_obs)).detach().cpu().numpy() for policy, next_obs in
                                               zip(self.target_policies, torch.swapaxes(next_obses, 0, 1))]).to(self.device)
                else:
                    target_actions = torch.Tensor([policy(next_obs).detach().cpu().numpy() for policy, next_obs in
                                                   zip(self.target_policies, torch.swapaxes(next_obses, 0, 1))]).to(self.device)
                target_actions = torch.swapaxes(target_actions, 0, 1)
                target_critic_in = torch.cat((next_obses, target_actions), dim=2).view(self.batch_size, -1)
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