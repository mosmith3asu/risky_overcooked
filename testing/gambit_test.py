import numpy as np
import pygambit as gbt
import time
n_games = 3
n_players = 2
Aflat =np.array([(i+1)*np.ones(6) for i in range(6)]).reshape(1,36).repeat(n_games,axis=0)
Bflat =np.array([(i+1)*np.ones(6) for i in range(6)]).reshape(1,36).repeat(n_games,axis=0)

games = np.zeros((n_games,n_players,6,6))
games[:,0,:,:] = Aflat.reshape(n_games,6,6)
games[:,1,:,:] = np.moveaxis(Bflat.reshape(n_games,6,6),-1,-2)

A = np.array([[0, -1, 1,0,-1], [-1, 0, 1,0,-1], [-1, 0, 1,0,-1],[-1, 0, 1,0,-1],[-1, 0, 1,0,-1]])
B = A.T


#
#
g = gbt.Game.from_arrays(A, B)
# g = gbt.Game.from_arrays(
#       [[1.1141, 0], [0, 0.2785]],
#       [[0, 1.1141], [1.1141, 0]])



tstart = time.time()
for _ in range(3):
      res =  gbt.nash.gnm_solve(g)
      # res = gbt.nash.logit_solve(g)
      # res =  gbt.nash.enummixed_solve(g,rational=False)
      # res =  gbt.nash.lcp_solve(g,rational=False)
      dist =  [profile[1] for profile in res.equilibria[0].mixed_strategies()]
      value = [res.equilibria[0].payoff(player) for player in ['1','2']]
print(f'Time: {time.time() - tstart}\t{dist}\t{value}')




tstart = time.time()
for _ in range(3):
      # res =  gbt.nash.gnm_solve(g)
      res = gbt.nash.logit_solve(g)
      # res =  gbt.nash.enummixed_solve(g,rational=False)
      # res =  gbt.nash.lcp_solve(g,rational=False)
      dist =  [profile[1] for profile in res.equilibria[0].mixed_strategies()]
      value = [res.equilibria[0].payoff(player) for player in ['1','2']]
print(f'Time: {time.time() - tstart}\t{dist}\t{value}')


tstart = time.time()
for _ in range(3):
      # res =  gbt.nash.gnm_solve(g)
      # res = gbt.nash.logit_solve(g)
      res =  gbt.nash.enummixed_solve(g,rational=False)
      # res =  gbt.nash.lcp_solve(g,rational=False)
      dist =  [profile[1] for profile in res.equilibria[0].mixed_strategies()]
      value = [res.equilibria[0].payoff(player) for player in ['1','2']]
print(f'Time: {time.time() - tstart}\t{dist}\t{value}')

tstart = time.time()
for _ in range(3):
      # res =  gbt.nash.gnm_solve(g)
      # res = gbt.nash.logit_solve(g)
      # res =  gbt.nash.enummixed_solve(g,rational=False)
      res =  gbt.nash.lcp_solve(g,rational=False,stop_after=1)
      dist =  [profile[1] for profile in res.equilibria[0].mixed_strategies()]
      value = [res.equilibria[0].payoff(player) for player in ['1','2']]
print(f'Time: {time.time() - tstart}\t{dist}\t{value}')


# print(list(g.mixed_strategy_profile()))
# # result = gbt.nash.enummixed_solve(g)
# # print(len(result.equilibria))
# # for eq in result.equilibria:
# #     print(lists(eq.mixed_strategies()))
# print(time.time()-tstart)