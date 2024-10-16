
import torch.nn as nn
from torch.optim import Adam
from src.risky_overcooked_rl.algorithms.MADDPG.utils import *

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

        self.lr = float(params.lr)
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
        self.critic = MLPNetwork(params.critic.input_dim, 1,
                                 hidden_dim=self.hidden_dim,
                                 constrain_out=False)
        self.target_policy = MLPNetwork(self.obs_dim, self.action_dim,
                                        hidden_dim=self.hidden_dim,
                                        constrain_out=constrain_out)
        self.target_critic = MLPNetwork(params.critic.input_dim, 1,
                                        hidden_dim=self.hidden_dim,
                                        constrain_out=False)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.lr * 0.1)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)

        self.exploration = OUNoise(self.action_dim)

        self.num_heads = 100

    def act(self, obs, explore=False):
        if obs.dim() == 1: obs = obs.unsqueeze(dim=0)
        action = self.policy(obs)
        if explore: action = gumbel_softmax(action, hard=True)
        else: action = onehot_from_logits(action)
        action = onehot_to_number(action)
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
    """Self-play MADDPG with shared policy and value functions"""

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

        # self.agent_index = params.agent_index
        self.num_agents = params.num_agents#len(self.agent_index)
        self.mse_loss = torch.nn.MSELoss()

        # Reshape critic input shape for shared observation
        params.critic.obs_dim = (self.obs_dim + self.action_dim) * self.num_agents
        self.agent = DDPGAgent(params).to(self.device)

    def scale_noise(self, scale):
        self.agent.scale_noise(scale)

    def reset_noise(self):
        self.agent.reset_noise()

    def act(self, observations, sample=False):
        if isinstance(observations, np.ndarray):
            observations = torch.Tensor(observations).to(self.device)
        self.agent.eval()
        actions = self.agent.act(observations, explore=sample).squeeze()
        self.agent.train()
        return np.array(actions)

    def update(self, replay_buffer):

        sample = replay_buffer.sample(self.batch_size)
        batch = replay_buffer.transition(*zip(*sample))
        BATCH_SIZE = len(sample)
        obs = torch.vstack(batch.obs)
        next_obs = torch.vstack(batch.next_obs)
        joint_action = torch.vstack(batch.action)
        reward = torch.vstack(batch.reward)
        done = torch.vstack(batch.done)

        # action = number_to_onehot(action)
        actions = torch.vstack([joint_action[:,0] // 6,joint_action[:,0] % 6]).swapaxes(0,1).unsqueeze(-1)
        actions = number_to_onehot(actions)
        actions = torch.hstack([actions[:,0,:],actions[:,1,:]])


        agent = self.agent
        target_policy = self.agent.target_policy

        ''' Update critic '''
        agent.critic_optimizer.zero_grad()
        with torch.no_grad():
            target_next_actions = torch.hstack([onehot_from_logits(target_policy(no)) for no in [next_obs, invert_obs(next_obs)]])
            target_critic_in = torch.hstack([next_obs, target_next_actions])
            target_next_q = reward + (1 - done) * self.gamma * agent.target_critic(target_critic_in)

        # critic_in = torch.cat((obses, actions), dim=2).view(self.batch_size, -1)
        critic_in = torch.hstack([obs, actions])
        main_q = agent.critic(critic_in)

        critic_loss = self.mse_loss(main_q, target_next_q)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        ''' Update policy '''
        agent.policy_optimizer.zero_grad()
        policy_out = agent.policy(obs)
        action = gumbel_softmax(policy_out, hard=True)
        partner_actions = onehot_from_logits(target_policy(invert_obs(obs)))
        target_actions = torch.hstack([action, partner_actions])

        critic_in = torch.hstack([obs, target_actions])
        actor_loss = -agent.critic(critic_in).mean()
        actor_loss += (policy_out ** 2).mean() * 1e-3  # Action regularize
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 0.5)
        agent.policy_optimizer.step()

        self.update_all_targets()



    def update_all_targets(self):
        soft_update(self.agent.target_critic, self.agent.critic, self.tau)
        soft_update(self.agent.target_policy, self.agent.policy, self.tau)


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
