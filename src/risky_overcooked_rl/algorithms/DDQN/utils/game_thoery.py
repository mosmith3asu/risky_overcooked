import numpy as np
import time
from numba import jit
# import matplotlib.pyplot as plt


# class QuantalResponseEQ:
#     def __init__(self, num_actions, num_states):
#
#
# def invert_game(g):
#     ego,partner = 0, 1
#     "inverts perspective of the game"
#     return np.hstack([np.transpose(g[:, partner, :, :], -1, -2).,
#                       np.transpose(g[:, ego, :, :], -1, -2)])
@jit
def level_k_qunatal(nf_games, rationality, sophistication=4, belief_trick=True):
    """Implementes a k-bounded QRE computation
    https://en.wikipedia.org/wiki/Quantal_response_equilibrium
    as reationality -> infty, QRE -> Nash Equilibrium
    """
    batch_sz = nf_games.shape[0]
    player_action_dim = nf_games.shape[2]
    uniform_dist = (np.ones([batch_sz, player_action_dim])) / player_action_dim
    ego, partner = 0, 1

    def invert_game(g):
        # Swap ego and partner and transpose the payoff matrices
        g_inv = np.zeros_like(g)
        g_inv[:, ego] = np.transpose(g[:, partner], axes=(0, 2, 1))
        g_inv[:, partner] = np.transpose(g[:, ego], axes=(0, 2, 1))
        return g_inv

    def softmax(x, axis=1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def step_QRE(game, k):
        if k == 0: partner_dist = uniform_dist
        else:  partner_dist = step_QRE(invert_game(game), k - 1)
        # Compute expected payoff for ego
        Exp_qAi = np.einsum('bij,bj->bi', game[:, ego], partner_dist)
        return softmax(rationality * Exp_qAi)

    def get_expected_equilibrium_value(games, dists):
        d_ego = dists[:, ego]
        d_partner = dists[:, partner]
        joint_dist_mat = np.einsum('bi,bj->bij', d_ego, d_partner)
        value_ego = np.sum(games[:, ego] * joint_dist_mat, axis=(1, 2), keepdims=True)
        value_partner = np.sum(games[:, partner] * joint_dist_mat, axis=(1, 2), keepdims=True)
        return np.concatenate([value_ego, value_partner], axis=1)

    dist1 = step_QRE(nf_games, sophistication)

    if belief_trick:
        # uses dist1 as partner belief prior for +1 sophistication
        Exp_qAi = np.einsum('bij,bj->bi', invert_game(nf_games)[:, ego], dist1)
        dist2 = softmax(rationality * Exp_qAi)
    else: # recomputes dist2 from scratch
        dist2 = step_QRE(invert_game(nf_games), sophistication)

    dist = np.stack([dist1, dist2], axis=1)
    # joint_dist_mat = torch.bmm(dist1.unsqueeze(-1), torch.transpose(dist2.unsqueeze(-1), -1, -2))

    value = get_expected_equilibrium_value(nf_games, dist)

    return dist, value


def main():
    batch_sz= 100_000
    num_agents =  2
    na = 6
    nf_games = np.random.rand(batch_sz, num_agents, na, na)

    start = time.time()
    level_k_qunatal(nf_games, 1.0, sophistication=4, belief_trick=True)
    print("Time taken:", time.time() - start)


if __name__ == "__main__":
    main()
