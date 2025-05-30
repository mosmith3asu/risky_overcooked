import random
from collections import namedtuple, deque
import torch
import numpy as np
# from risky_overcooked_rl.algorithms.DDQN.utils import invert_obs, invert_joint_action, invert_prospect
from risky_overcooked_rl.utils.state_utils import invert_obs, invert_joint_action, invert_prospect
# REPLAY MEMORY ----------------

class ReplayMemory_Simple(object):
    " For training rational response (partner/robot) agent"

    def __init__(self, capacity,device,with_priority=False):
        self.memory = deque([], maxlen=capacity)
        self.transition = namedtuple('Transition', ('state', 'action', 'reward','next_state','done'))
        self.device = device

        self.with_priority = with_priority
        if self.with_priority:
            self.tree = SumTree(size=capacity)

            # PER params
            self.eps = 1e-2  # minimal priority, prevents zero probabilities
            self.alpha = 0.1  # determines how much prioritization is used, α = 0 corresponding to the uniform case
            self.beta = 0.1  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
            self.max_priority = self.eps  # priority for new samples, init as eps

            # self.count = 0
            # self.real_size = 0
            # self.size = capacity


    def push(self, state, action, reward, next_state,done):
        """Save a transition ONLY FOR ONE AGENT"""
        if not isinstance(action, torch.Tensor): action = torch.tensor(action, dtype=torch.int64, device=self.device).reshape(1, 1).to(self.device)
        if not isinstance(done, torch.Tensor): done = torch.tensor(done, dtype=torch.int64, device=self.device).reshape(1, 1).to(self.device)
        if not isinstance(reward, torch.Tensor): reward = torch.tensor(reward, dtype=torch.float32, device=self.device).reshape(1, 1).to(self.device)
        args = (state, action, reward, next_state, done)
        self.memory.append(self.transition(*args))

        if self.with_priority:
            # store transition index with maximum priority in sum tree
            self.tree.add(self.max_priority, len(self)-1)

            # update counters
            # self.count = (self.count + 1) % self.size
            # self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def priority_sample(self, batch_size):
        # assert len(self) >= batch_size, "buffer contains less samples than batch size"

        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = float(priority)
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)
        probs = priorities / self.tree.total

        # The estimation of the expected value with stochastic updates relies on those updates corresponding
        # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
        # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
        # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
        # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
        # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
        # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).
        # For stability reasons, we always normalize weights by 1/maxi wi so that they only scale the
        # update downwards (Section 3.4, first paragraph)
        weights = (len(self) * probs) ** -self.beta

        # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
        # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
        # within a reasonable range, avoiding the possibility of extremely large updates. (Appendix B.2.1, Proportional prioritization)
        weights = weights / weights.max()

        # batch = self.memory[np.array(sample_idxs)]
        batch = [self.memory[i] for i in sample_idxs]

        return batch, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)


class ReplayMemory_Prospect(object):
    """
    Prioritization Source:
    https://paperswithcode.com/method/prioritized-experience-replay
    """

    def __init__(self, capacity, device,with_priority=False):
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_prospects', 'done'))
        self.memory = deque([], maxlen=capacity)
        self.device = device

        self.with_priority = with_priority
        if self.with_priority:
            self.tree = SumTree(size=capacity)

            # PER params
            self.eps = 1e-2  # minimal priority, prevents zero probabilities
            self.alpha = 0.1  # determines how much prioritization is used, α = 0 corresponding to the uniform case
            self.beta = 0.1  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
            self.max_priority =  self.eps  # priority for new samples, init as eps

            # self.count = 0
            # self.real_size = 0
            # self.size = capacity


    # def double_push(self, state, action, rewards, next_prospects, done):
    #
    #     """ Push both agent's experience into memory from ego perspective"""
    #     if not isinstance(action, torch.Tensor):
    #         action = torch.tensor(action, dtype=torch.int64, device=self.device).reshape(1, 1).to(self.device)
    #     # if not isinstance(done, torch.Tensor): done = torch.tensor(done, dtype=torch.int64, device=self.device).reshape(1, 1).to(self.device)
    #     rewards = rewards.flatten()
    #
    #     # Append Agent 1 experiences
    #     # reward = torch.tensor([rewards[0]], dtype=torch.float32, device=self.device).reshape(1, 1).to(self.device)
    #     reward = rewards
    #
    #     # assert len(reward.shape)==2,f'reward shape should be 2D:{reward.shape}'
    #     # self.memory.append(self.transition(state, action, reward, next_prospects, done))
    #     self.push(state, action, reward, next_prospects, done)

    def double_push(self, state, action, rewards, next_prospects, done):

        """ Push both agent's experience into memory from ego perspective"""

        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.int64, device=self.device).reshape(1, 1).to(self.device)
        # if not isinstance(done, torch.Tensor): done = torch.tensor(done, dtype=torch.int64, device=self.device).reshape(1, 1).to(self.device)
        rewards = rewards.flatten()

        # Append Agent 1 experiences
        # reward = torch.tensor([rewards[0]], dtype=torch.float32, device=self.device).reshape(1, 1).to(self.device)
        reward = rewards[0]

        # assert len(reward.shape)==2,f'reward shape should be 2D:{reward.shape}'
        # self.memory.append(self.transition(state, action, reward, next_prospects, done))
        self.push(state, action, reward, next_prospects, done)

        # # Append Agent 2 experience
        s_prime = invert_obs(state)
        a_prime = invert_joint_action(action).to(self.device)
        # r_prime = torch.tensor([rewards[1]], dtype=torch.float32, device=self.device).reshape(1, 1).to(self.device)
        r_prime = rewards[1]
        np_prime = invert_prospect(next_prospects)
        self.push(s_prime, a_prime, r_prime, np_prime, done)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.transition(*args))

        if self.with_priority:
            # store transition index with maximum priority in sum tree
            self.tree.add(self.max_priority, len(self)-1)

            # update counters
            # self.count = (self.count + 1) % self.size
            # self.real_size = min(self.size, self.real_size + 1)


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def priority_sample(self, batch_size):
        # assert len(self) >= batch_size, "buffer contains less samples than batch size"

        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = float(priority)
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)
        probs = priorities / self.tree.total

        # The estimation of the expected value with stochastic updates relies on those updates corresponding
        # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
        # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
        # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
        # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
        # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
        # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).
        # For stability reasons, we always normalize weights by 1/maxi wi so that they only scale the
        # update downwards (Section 3.4, first paragraph)
        weights = (len(self) * probs) ** -self.beta

        # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
        # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
        # within a reasonable range, avoiding the possibility of extremely large updates. (Appendix B.2.1, Proportional prioritization)
        weights = weights / weights.max()

        # batch = self.memory[np.array(sample_idxs)]
        batch = [self.memory[i] for i in sample_idxs]

        return batch, weights, tree_idxs
    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.memory)
# class ReplayMemory_Prospect(object):
#
#     def __init__(self, capacity, device):
#         self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_prospects', 'done'))
#         self.memory = deque([], maxlen=capacity)
#         self.device = device
#
#     def double_push(self, state, action, rewards, next_prospects, done):
#         """ Push both agent's experience into memory from ego perspective"""
#         if not isinstance(action, torch.Tensor): action = torch.tensor(action, dtype=torch.int64, device=self.device).reshape(1, 1).to(self.device)
#         if not isinstance(done, torch.Tensor): done = torch.tensor(done, dtype=torch.int64, device=self.device).reshape(1, 1).to(self.device)
#         rewards = rewards.flatten()
#
#         # Append Agent 1 experiences
#         reward = torch.tensor([rewards[0]], dtype=torch.float32, device=self.device).reshape(1, 1).to(self.device)
#
#         # assert len(reward.shape)==2,f'reward shape should be 2D:{reward.shape}'
#         self.memory.append(self.transition(state, action, reward, next_prospects, done))
#
#         # # Append Agent 2 experience
#         s_prime = invert_obs(state)
#         a_prime = invert_joint_action(action).to(self.device)
#         r_prime = torch.tensor([rewards[1]], dtype=torch.float32, device=self.device).reshape(1, 1).to(self.device)
#         np_prime = invert_prospect(next_prospects)
#         # assert len(r_prime.shape) == 2, f'reward shape should be 2D:{r_prime.shape}'
#         self.memory.append(self.transition(s_prime, a_prime, r_prime, np_prime, done))
#
#     def push(self, *args):
#         """Save a transition"""
#         self.memory.append(self.transition(*args))
#
#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)
#
#     def __len__(self):
#         return len(self.memory)
class SumTree:
    def __init__(self, size):
        self.nodes = [0] * (2 * size - 1)
        # self.data = [None] * size
        self.data = deque([], maxlen=size)
        self.size = size
        # self.count = 0
        # self.real_size = 0

    @property
    def total(self):
        return self.nodes[0]

    def update(self, data_idx, value):
        idx = data_idx + self.size - 1  # child index in tree array
        change = value - self.nodes[idx]

        self.nodes[idx] = value

        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value, data):
        # self.data[self.count] = data
        # self.update(self.count, value)
        self.data.append(data)
        self.update(len(self.data)-1, value)

        # self.count = (self.count + 1) % self.size
        # self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum):
        assert cumsum <= self.total

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2*idx + 1, 2*idx + 2

            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.nodes[left]

        data_idx = idx - self.size + 1

        return data_idx, self.nodes[idx], self.data[data_idx]

    def __repr__(self):
        return f"SumTree(nodes={self.nodes.__repr__()}, data={self.data.__repr__()})"

# class ReplayMemory_Simple(object):
#     " For training rational response (partner/robot) agent"
#
#     def __init__(self, capacity,device):
#         self.memory = deque([], maxlen=capacity)
#         self.transition = namedtuple('Transition', ('state', 'action', 'reward','next_state','done'))
#         self.device = device
#         self.with_priority = False
#
#     def push(self, state, action, reward, next_state,done):
#         """Save a transition ONLY FOR ONE AGENT"""
#         if not isinstance(action, torch.Tensor): action = torch.tensor(action, dtype=torch.int64, device=self.device).reshape(1, 1).to(self.device)
#         if not isinstance(done, torch.Tensor): done = torch.tensor(done, dtype=torch.int64, device=self.device).reshape(1, 1).to(self.device)
#         if not isinstance(reward, torch.Tensor): reward = torch.tensor(reward, dtype=torch.float32, device=self.device).reshape(1, 1).to(self.device)
#         args = (state, action, reward, next_state, done)
#         self.memory.append(self.transition(*args))
#
#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)
#
#     def __len__(self):
#         return len(self.memory)
#
#     def update_priorities(self, data_idxs, priorities):
#         raise NotImplementedError
#     def priority_sample(self, batch_size):
#         raise NotImplementedError