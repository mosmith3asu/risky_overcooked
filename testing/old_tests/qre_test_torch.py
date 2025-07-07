import numpy as np
import torch



def get_normal_form_game(Aflat,Bflat):
    num_agents = 2
    player_action_dim = 6
    batch_size = Aflat.shape[0]
    all_games = torch.zeros([batch_size, num_agents, player_action_dim, player_action_dim])
    for i in range(num_agents):
        q_values = Aflat if i==0 else Bflat#TODO: REPLACE!!!
        q_values = q_values.reshape(batch_size, player_action_dim, player_action_dim)
        all_games[:, i, :, :] = q_values if i==0 else torch.transpose(q_values,-1,-2)
    return all_games

def level_k_qunatal_torch(nf_games, k=8, rationality=10):
    """Implementes a k-bounded QRE computation
    https://en.wikipedia.org/wiki/Quantal_response_equilibrium
    as reationality -> infty, QRE -> Nash Equilibrium
    """
    num_players = nf_games.shape[1]
    batch_sz = nf_games.shape[0]
    player_action_dim = nf_games.shape[2]
    uniform_dist = (torch.ones(batch_sz,player_action_dim))/ player_action_dim #TODO: DEVICE!!
    ego,partner =0,1
    def invert_game(g):
        "inverts perspective of the game"
        return torch.cat([  torch.transpose(g[:, partner, :, :], -1, -2).unsqueeze(1),
                            torch.transpose(g[:, ego, :, :], -1, -2).unsqueeze(1)], dim=1)

    def softmax(x):
        ex = torch.exp(x - torch.max(x,dim=1).values.reshape(-1,1).repeat(1,player_action_dim))
        return ex/torch.sum(ex,dim=1).reshape(-1,1).repeat(1,player_action_dim)
        # ex = np.exp(x - np.max(x))
        # return ex / np.sum(ex)

    def step_QRE(game, k):
        if k == 0:
            partner_dist = uniform_dist
        else:
            partner_dist = step_QRE(invert_game(game), k - 1)
        Exp_qAi = torch.bmm(game[:,ego,:,:],partner_dist.unsqueeze(-1)).squeeze(-1)
        return softmax(rationality * Exp_qAi)

    dist1 = step_QRE(nf_games, k)
    dist2 = step_QRE(invert_game(nf_games), k)
    dist = torch.cat([dist1.unsqueeze(1),dist2.unsqueeze(1)],dim=1)
    # dist = [list(dist1), list(dist2)]
    # dist_mat = dist1.reshape(player_action_dim, 1) @ dist2.reshape(1, player_action_dim)
    joint_dist_mat = torch.bmm(dist1.unsqueeze(-1), torch.transpose(dist2.unsqueeze(-1), -1, -2))
    value = [torch.sum(nf_games[:,ego, :] * joint_dist_mat,dim=(-1,-2)),
             torch.sum(nf_games[:,partner, :] * joint_dist_mat,dim=(-1,-2))]
    return dist, value

def invert_game(g):
    "inverts perspective of the game"
    # return np.array([g[1, :].T, g[0, :].T])
    # return np.array([g[1, :].T, g[0, :].T])
    return torch.cat([torch.transpose(g[:,1,:,:],-1,-2).unsqueeze(1),torch.transpose(g[:,0,:,:],-1,-2).unsqueeze(1)],dim=1)

def softmax(x):
    ex = np.exp(x-np.max(x))
    return ex/np.sum(ex)


def step_QRE(game,k):
    if k==0:
        na = np.shape(game)[1]
        # partner_dist = (np.arange(np.shape(game)[1])).reshape(1, np.shape(game)[1])
        # partner_dist = (np.ones(na)).reshape(1, na) / na
        (torch.ones(na)).reshape(1, na) / na
    else:
        inv_game = np.array([game[1, :].T, game[0, :].T ])
        partner_dist = step_QRE(inv_game,k-1)
    weighted_game = game[0] * partner_dist
    Exp_qAi = np.sum(weighted_game, axis=1)
    return softmax(Exp_qAi)



# def step_QRE(game,k):
#     if k==0:
#         na = np.shape(game)[1]
#         # partner_dist = (np.arange(np.shape(game)[1])).reshape(1, np.shape(game)[1])
#         partner_dist = (np.ones(na)).reshape(1, na) / na
#     else:
#         inv_game = np.array([game[1, :].T, game[0, :].T ])
#         partner_dist = step_QRE(inv_game,k-1)
#     weighted_game = game[0] * partner_dist
#     Exp_qAi = np.sum(weighted_game, axis=1)
#     return softmax(Exp_qAi)
#
#



def main():
    A = (np.array([i * np.ones(6) for i in range(6)])).flatten().reshape(1,-1).repeat(3,0)
    B = (np.array([i * np.ones(6) for i in range(6)])).flatten().reshape(1,-1).repeat(3,0)

    Aflat = torch.FloatTensor(A)
    Bflat = torch.FloatTensor(B)
    nf_game = get_normal_form_game(Aflat,Bflat)
    # invert_game(nf_game)
    dist, value = level_k_qunatal_torch(nf_game)
    # A = (np.array([i*np.ones(6) for i in range(6)])).reshape(1,6,6)
    # B = (np.array([i*np.ones(6) for i in range(6)]).T).reshape(1,6,6)
    # game = torch.FloatTensor(np.vstack([A, B]))
    # game = invert_game(game)
    # print(game)
    dist, value = level_k_qunatal_torch(game, k=0)
    # A = np.zeros([1, 6, 6])
    # A[0, 1, :] = 1000
    # A[0, 2, :] = 1000
    # A[0, :, 1] = 0
    #
    # B = np.zeros([1, 6, 6])
    # B[0, :, 1] = 1000
    # B[0, :, 1] = 1000

    # A = np.repeat(np.arange(6).reshape(1,6),6,axis=0).reshape([1,6,6])
    # B = np.repeat(np.arange(6).reshape(1,6).T,6,axis=1).reshape([1,6,6])
    # game = np.vstack([A,B])
    # game = torch.FloatTensor(game)



    print(np.all(invert_game(invert_game(game))==game))
    print(game.shape)
    print(A)
    print(B)
    na = np.shape(game)[1]

    inv_game = np.array([game[1,:].T,game[0,:].T,])
    print(inv_game.shape)

    # pdj = (np.arange(na)).reshape(1,na)
    # weighted_game = game[0]*pdj
    # Exp_qAi = np.sum(weighted_game,axis=1)
    # print(Exp_qAi)
    dist1 = step_QRE(game, k=5).reshape(na,1)
    dist2 = step_QRE(invert_game(game), k=5).reshape(1,na)
    dist_mat = dist1@dist2
    print('dist1',dist1)
    print('dist2', dist2.flatten())
    print('mat:',dist_mat)

    # print(step_QRE(game,k=5))
    # print(step_QRE(invert_game(game), k=5))
    # sum of ego agent is row
    # qsumi = np.sum(game,axis=1)[0]
    # qAi_given_pdj = pdj*np.sum(game,axis=-1)[0]
    #
    # Exp_qAi_given_pdj = pdj*np.sum(game,axis=-1)[0]
    # print(qAi_given_pdj)
if __name__ == "__main__":
    main()