import numpy as np

def softmax(x):
    ex = np.exp(x-np.max(x))
    return ex/np.sum(ex)

# def step_QRE(game,partner_dist):
#     weighted_game = game[0] * partner_dist
#     Exp_qAi = np.sum(weighted_game, axis=1)
#     return softmax(Exp_qAi)
def step_QRE(game,k):
    if k==0:
        partner_dist = (np.arange(np.shape(game)[1])).reshape(1,np.shape(game)[1])
    else:
        inv_game = np.array([game[1, :].T, game[0, :].T ])
        partner_dist = step_QRE(inv_game,k-1)
    weighted_game = game[0] * partner_dist
    Exp_qAi = np.sum(weighted_game, axis=1)
    return softmax(Exp_qAi)
def main():
    # A = np.ones([1,6,6])
    # B = 2*np.ones([1,6,6])
    A = np.zeros([1, 6, 6])
    A[0, 1, :] = 1000
    A[0, 2, 0] = 1000

    B = np.zeros([1, 6, 6])
    B[0, :, 1] = 1000
    # B[0, :, 1] = 1000

    # A = np.repeat(np.arange(6).reshape(1,6),6,axis=0).reshape([1,6,6])
    B = np.repeat(np.arange(6).reshape(1,6).T,6,axis=1).reshape([1,6,6])
    game = np.vstack([A,B])
    # print(game.shape)
    # print(A)
    # print(B)
    na = np.shape(game)[1]

    inv_game = np.array([game[1,:].T,game[0,:].T,])
    print(inv_game.shape)

    # pdj = (np.arange(na)).reshape(1,na)
    # weighted_game = game[0]*pdj
    # Exp_qAi = np.sum(weighted_game,axis=1)
    # print(Exp_qAi)

    print(step_QRE(game,k=5))
    # sum of ego agent is row
    # qsumi = np.sum(game,axis=1)[0]
    # qAi_given_pdj = pdj*np.sum(game,axis=-1)[0]
    #
    # Exp_qAi_given_pdj = pdj*np.sum(game,axis=-1)[0]
    # print(qAi_given_pdj)
if __name__ == "__main__":
    main()