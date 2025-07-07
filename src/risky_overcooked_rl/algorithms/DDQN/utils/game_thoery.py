import numpy as np
import time
import torch
import itertools
from risky_overcooked_py.mdp.actions import Action
from numba import jit,njit
import numba
# from numba import float32, int32

class QuantalResponse_torch:
    def __init__(self,rationality,sophistication,joint_action_space, belief_trick=True,**kwargs):
        self.player_action_dim = len(Action.ALL_ACTIONS)
        self.num_agents = 2
        self.ego,self.partner= 0, 1
        self.joint_action_space = joint_action_space
        self.device = torch.cuda.current_device() #if torch.cuda.is_available() else torch.device('cpu')

        self.rationality = rationality       # rationality/temperature parameter in softmax
        self.sophistication = sophistication # number of recursive steps in QRE computation
        self.belief_trick=True               # reduces computation by using the first distribution as a prior for the second

        self.uniform_dist = (torch.ones(256, self.player_action_dim, device=self.device))/ self.player_action_dim


    def get_expected_equilibrium_value(self, NF_games, dists):
        """ Computes expected value for both agents following the equilibrium strategy."""
        joint_dist_mat = torch.bmm(dists[:, self.ego].unsqueeze(-1),
                                   torch.transpose(dists[:, self.partner].unsqueeze(-1), -1, -2))
        value = torch.cat([torch.sum(NF_games[:, self.ego, :] * joint_dist_mat, dim=(-1, -2)).unsqueeze(-1),
                           torch.sum(NF_games[:, self.partner, :] * joint_dist_mat, dim=(-1, -2)).unsqueeze(-1)], dim=1)
        return value
    def invert_game(self, g):
            "inverts perspective of the game"
            return torch.cat([torch.transpose(g[:, self.partner, :, :], -1, -2).unsqueeze(1),
                              torch.transpose(g[:, self.ego, :, :], -1, -2).unsqueeze(1)], dim=1)

    def softmax(self, x):
            return torch.softmax(x,dim=1)

    def step_QRE(self, game, k):
            if k == 0:  partner_dist = self.uniform_dist
            else:  partner_dist = self.step_QRE(self.invert_game(game), k - 1)

            Exp_qAi = torch.bmm(game[:, self.ego, :, :], partner_dist.unsqueeze(-1)).squeeze(-1)
            return self.softmax(self.rationality * Exp_qAi)

    def level_k_qunatal(self,nf_games, sophistication=4):
        """Implementes a k-bounded QRE computation
        https://en.wikipedia.org/wiki/Quantal_response_equilibrium
        as reationality -> infty, QRE -> Nash Equilibrium
        """
        rationality = self.rationality
        batch_sz = nf_games.shape[0] # batch of all prospects, not defined samples
        player_action_dim = nf_games.shape[2]
        self.uniform_dist = (torch.ones(batch_sz, player_action_dim,device=self.device)) / player_action_dim
        ego, partner = 0, 1

        dist1 = self.step_QRE(nf_games, sophistication)
        if self.belief_trick:
            # uses dist1 as partner belief prior for +1 sophistication
            Exp_qAi = torch.bmm(self.invert_game(nf_games)[:, ego, :, :],dist1.unsqueeze(-1)).squeeze(-1)
            dist2 = self.softmax(rationality * Exp_qAi)
        else: # recomputes dist2 from scratch
            dist2 = self.step_QRE(self.invert_game(nf_games), sophistication)
        dist = torch.cat([dist1.unsqueeze(1), dist2.unsqueeze(1)], dim=1)
        value = self.get_expected_equilibrium_value(nf_games, dist)
        return dist, value


    def compute_EQ(self, NF_games):
        NF_Games = NF_games.reshape(-1, self.num_agents, self.player_action_dim, self.player_action_dim)
        all_dists, all_ne_values = self.level_k_qunatal(NF_Games)
        # if update:
        #     return all_dists, all_ne_values
        return all_dists, all_ne_values

    def choose_actions(self,NF_games):
        """Sample actions from strategies"""
        action_probs, all_ne_values = self.compute_EQ(NF_games)

        # New cpu version
        action_probs = action_probs.cpu().detach().numpy()
        all_joint_actions = []
        for prob in action_probs:
            a1 = np.random.choice(len(prob[0]), p=prob[0])  # Sample action for agent 1
            a2 = np.random.choice(len(prob[1]), p=prob[1])  # Sample action for agent 2
            action_idxs = (a1, a2)
            joint_action_idx = Action.INDEX_TO_ACTION_INDEX_PAIRS.index(action_idxs)
            all_joint_actions.append(joint_action_idx)
        joint_action_idx = np.array(all_joint_actions)[0]
        joint_action = self.joint_action_space[joint_action_idx]
        return joint_action, joint_action_idx, action_probs

        # all_joint_actions = []
        # for i in action_probs:
        #     a1, a2 = torch.multinomial(action_probs[0, :], 1).detach().cpu().numpy().flatten()
        #     action_idxs = (a1, a2)
        #     joint_action_idx = Action.INDEX_TO_ACTION_INDEX_PAIRS.index(action_idxs)
        #     all_joint_actions.append(joint_action_idx)
        # joint_action_idx = np.array(all_joint_actions)[0]
        # joint_action = self.joint_action_space[joint_action_idx]
        # return joint_action, joint_action_idx, action_probs




# @njit
def level_k_qunatal(nf_games, rationality, sophistication=4, belief_trick=True):
    """Implementes a k-bounded QRE computation
    https://en.wikipedia.org/wiki/Quantal_response_equilibrium
    as reationality -> infty, QRE -> Nash Equilibrium
    """
    batch_sz = nf_games.shape[0]
    player_action_dim = nf_games.shape[2]
    uniform_dist = (np.ones((batch_sz, player_action_dim),dtype=np.float64)) / player_action_dim


    ego, partner = 0, 1

    def invert_game(g):
        # Swap ego and partner and transpose the payoff matrices
        g_inv = np.zeros_like(g)
        g_inv[:, ego] = np.transpose(g[:, partner], axes=(0, 2, 1))
        g_inv[:, partner] = np.transpose(g[:, ego], axes=(0, 2, 1))
        return g_inv

    def softmax(x, axis=1):
        # e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        # return e_x / np.sum(e_x, axis=axis, keepdims=True)
        _maxi = np.array([np.max(x[i,:]) for i in range(x.shape[0])])[:,np.newaxis]
        e_x = np.exp(x - _maxi)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)


    # @jit(["float32[:,:]", "int32(int32)"], nopython=True)
    # def step_QRE(game, k):
    #     if k == 0: partner_dist = uniform_dist
    #     else:  partner_dist = step_QRE(invert_game(game), k - 1)
    #     # Compute expected payoff for ego
    #     Exp_qAi = np.einsum('bij,bj->bi', game[:, ego], partner_dist)
    #     return softmax(rationality * Exp_qAi)

    def get_expected_equilibrium_value(games, dists):
        d_ego = dists[:, ego]
        d_partner = dists[:, partner]
        joint_dist_mat = np.einsum('bi,bj->bij', d_ego, d_partner)
        value_ego = np.sum(games[:, ego] * joint_dist_mat, axis=(1, 2), keepdims=True)
        value_partner = np.sum(games[:, partner] * joint_dist_mat, axis=(1, 2), keepdims=True)
        return np.concatenate([value_ego, value_partner], axis=1)


    # Rollout QRE Recursion ###########################################
    # dist1 = step_QRE(nf_games, sophistication)
    partner_pA = uniform_dist
    Gi = nf_games if sophistication % 2 == 0 else invert_game(nf_games)

    for k in range(sophistication):
        # Compute expected payoff for ego
        # Exp_qAi = np.einsum('bij,bj->bi', Gi[:, ego], partner_pA)
        Exp_qAi = np.sum(Gi[:, ego] * partner_pA[:, :, np.newaxis], axis=-1)
        ego_pA = softmax(rationality * Exp_qAi)

        # Change perspective
        partner_pA = ego_pA
        Gi = invert_game(Gi)

    dist1 = ego_pA


    if belief_trick:
        # uses dist1 as partner belief prior for +1 sophistication
        # Exp_qAi = np.einsum('bij,bj->bi', Gi[:, ego], partner_pA)
        Exp_qAi = np.sum(Gi[:, ego] * partner_pA[:, :, np.newaxis], axis=-1)
        dist2 = softmax(rationality * Exp_qAi)
    else: # recomputes dist2 from scratch
        partner_pA = uniform_dist
        Gi = invert_game(nf_games)  if sophistication % 2 == 0 else nf_games
        for k in range(sophistication):
            # Compute expected payoff for ego
            # Exp_qAi = np.einsum('bij,bj->bi', Gi[:, ego], partner_pA)
            Exp_qAi = np.sum(Gi[:, ego] * partner_pA[:, :, np.newaxis], axis=-1)
            ego_pA = softmax(rationality * Exp_qAi)
            # Change perspective
            partner_pA = ego_pA
            Gi = invert_game(Gi)
        dist2 = ego_pA

    # dist = np.stack([dist1, dist2], axis=1)
    dist = np.column_stack([dist1[:,np.newaxis,:], dist2[:,np.newaxis,:]])

    value = get_expected_equilibrium_value(nf_games, dist)

    return dist, value
#
# @jit(nopython=True)
# def f():
#     return np.zeros([5, 12], dtype=np.float32)

def main():
    n_trials = 1000000
    batch_sz= 256
    num_agents =  2
    na = 6
    nf_games = np.random.rand(batch_sz, num_agents, na, na)
    level_k_qunatal(nf_games, 1.0, sophistication=4, belief_trick=True) # compile



    nf_games_torch = torch.from_numpy(nf_games).float().cuda()  # Ensure the tensor is on GPU if available
    qr = QuantalResponse_torch(rationality=1.0, sophistication=4, joint_action_space=36, belief_trick=True)

    start = time.time()
    for i in range(n_trials):
        level_k_qunatal(nf_games, 1.0, sophistication=4, belief_trick=True)
    print("Time taken (numba):", time.time() - start)

    start = time.time()
    for i in range(n_trials):
        qr.level_k_qunatal(nf_games_torch)
    print("Time taken (torch):", time.time() - start)
    #
    # qr.compute_EQ(nf_games_torch)
    #
    # qr.level_k_qunatal(nf_games_torch)
    # print("Time taken:", time.time() - start)


if __name__ == "__main__":
    main()
