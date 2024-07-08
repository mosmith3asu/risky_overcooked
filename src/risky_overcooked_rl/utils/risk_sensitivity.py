import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class CumulativeProspectTheory(object):
    def __init__(self,b,lam,eta_p,eta_n,delta_p,delta_n):
        """
        :param b: reference point determining if outcome is gain or loss
        :param lam: loss-aversion parameter
        :param eta_p: exponential gain on positive outcomes
        :param eta_n: exponential loss on negative outcomes
        :param delta_p: probability weighting for positive outcomes
        :param delta_n: probability weighting for negative outcomes
        """
        self.track_value_history = isinstance(b, str)

        self.b = b if isinstance(b, (int, float)) else 0
        self.lam = lam
        self.eta_p = eta_p
        self.eta_n = eta_n
        self.delta_p = delta_p
        self.delta_n = delta_n

        self.value_history = deque(maxlen=1000)
        self.value_history.append(0)

    def recalc_b(self):
        # TODO: this might be very inefficient this way
        self.b = np.mean(self.value_history)
        return self.b

    def expectation_PT(self, values, p_values):
        if self.track_value_history:
            self.value_history.extend(values)

        # arrange all samples in ascending order
        vp = values[np.where(values >= self.b)[0]]
        vn = values[np.where(values <= self.b)[0]]
        u_plus = self.u_plus(vp)
        u_neg = -1 * self.u_neg(vn)
        u = np.hstack([u_neg, u_plus])


        p = np.linspace(0,1,100)
        pp = p_values[np.where(values >= self.b)[0]]
        pn = p_values[np.where(values <= self.b)[0]]
        w_plus = self.w_plus(pp)
        w_neg = self.w_neg(pn)
        w = np.hstack([w_neg, w_plus])
        rho = np.sum(u*w)
        return rho

    def expectation(self, values, p_values):
        if self.track_value_history:
            self.value_history.extend(values)

        # arrange all samples in ascending order
        sorted_idxs = np.argsort(values)
        sorted_v = values[sorted_idxs]
        sorted_p = p_values[sorted_idxs]
        K = len(sorted_v)  # number of samples

        # If there is only one value, return the utility of that value
        if K==1:
            if sorted_v[0]>self.b: return self.u_plus(sorted_v[0])
            else: return -1*self.u_neg(sorted_v[0])
        elif np.all(sorted_v<=self.b):
            Fk = [np.sum(sorted_p[0:i + 1]) for i in range(K)]
            l=K-1
            rho_p = self.rho_neg(sorted_v, sorted_p, Fk, l, K)
            rho_n = 0
            rho = rho_p - rho_n
            return rho
        elif np.all(sorted_v > self.b):
            Fk = [np.sum(sorted_p[i:K]) for i in range(K)]
            l = 0
            rho_p = 0
            rho_n = self.rho_neg(sorted_v, sorted_p, Fk, l, K)
            rho = rho_p - rho_n
            return rho
        else:
            l = np.where(sorted_v <= self.b)[0][-1]  # idx of highest loss
            Fk = [np.sum(sorted_p[0:i + 1]) for i in range(l + 1)] + \
                 [np.sum(sorted_p[i:K]) for i in range(l + 1, K)]  # cumulative probability
            rho_p = self.rho_plus(sorted_v, sorted_p, Fk, l, K)
            rho_n = self.rho_neg(sorted_v, sorted_p, Fk, l, K) if l >= 0 else 0
            rho = rho_p - rho_n
            return rho

    def rho_plus(self,sorted_v,sorted_p,Fk,l,K):
        rho_p = 0
        for i in range(l + 1, K - 1):
            rho_p += self.u_plus(sorted_v[i]) * (self.w_plus(Fk[i]) - self.w_plus(Fk[i - 1]))
        rho_p += self.u_plus(sorted_v[K - 1]) * self.w_plus(sorted_p[K - 1])
        return rho_p
    def rho_neg(self,sorted_v,sorted_p,Fk,l,K):
        rho_n = self.u_neg(sorted_v[0]) * self.w_neg(sorted_p[0])
        for i in range(1, l + 1):
            rho_n += self.u_neg(sorted_v[i]) * (self.w_neg(Fk[i]) - self.w_neg(Fk[i - 1]))
        return rho_n
    def u_plus(self,v):
        return np.abs(v-self.b)**self.eta_p
    def u_neg(self, v):
        # return -1*self.lam * np.abs(v-self.b) ** self.eta_n
        return self.lam * np.abs(v-self.b) ** self.eta_n

    def w_plus(self,p):
        delta = self.delta_p
        return p**delta/((p**delta + (1-p)**delta)**(1/delta))
    def w_neg(self,p):
        delta = self.delta_n
        return p**delta/((p**delta + (1-p)**delta)**(1/delta))

    def plot_curves(self):
        fig,axs = plt.subplots(1,2,constrained_layout=True)

        # Plot utility functions
        v = np.linspace(-10,10,100)
        vp = v[np.where(v>=self.b)[0]]
        vn = v[np.where(v<=self.b)[0]]
        u_plus = self.u_plus(vp)
        u_neg = -1*self.u_neg(vn)
        u = np.hstack([u_neg,u_plus])
        # axs[0].plot(vp,u_plus,color='r')
        # axs[0].plot(vn,u_neg,color='b')
        axs[0].plot(v, u, color='g')
        axs[0].plot(v, v, color='gray',linestyle='--')
        axs[0].set_xlabel('Reward')
        axs[0].set_ylabel('Perceived Reward')
        axs[0].plot([v[0], v[-1]], [0, 0], c="lightgrey", zorder=1, lw=1)
        axs[0].plot([0, 0], [v[0], v[-1]], c="lightgrey", zorder=1, lw=1)
        #make axis square
        axs[0].set_aspect('equal', adjustable='box')

        # Plot probability weighting functions
        p = np.linspace(0,1,100)
        pp = self.w_plus(p)
        pn = self.w_neg(p)
        axs[1].plot(p, pp,color='r',label='$w^+$')
        axs[1].plot(p, pn,color='b',label='$w^-$')
        axs[1].plot(p, p, color='gray', label='Rational',linestyle='--')
        axs[1].set_xlabel('Probability')
        axs[1].set_ylabel('Decision Weight')
        axs[1].legend(frameon=False,ncol=1)
        axs[1].set_aspect('equal', adjustable='box')
        #
        #
        # w_plus = self.w_plus(v)
        # w_neg = self.w_neg(v)
        #
        # plt.plot(v,u_plus,label='u_plus')
        # plt.plot(v,u_neg,label='u_neg')
        # plt.plot(v,w_plus,label='w_plus')
        # plt.plot(v,w_neg,label='w_neg')
        # plt.legend()
        plt.show()

def main():
    # example from https://engineering.purdue.edu/DELP/education/decision_making_slides/Module_12___Cumulative_Prospect_Theory.pdf
    values = np.array([80,60,40,20,0])
    p_values = np.array([0.2,0.2,0.2,0.2,0.2])

    # values = np.array([-1,1])
    # p_values = np.ones(len(values)) / len(values)
    # cpt_params = {'b':2.0, 'lam':2.0,
    #               'eta_p':0.88,'eta_n':0.5,
    #               'delta_p':0.88,'delta_n':0.6}
    cpt_params = {'b': 0, 'lam': 1.0,
                  'eta_p': 1., 'eta_n': 1.,
                  'delta_p': 1., 'delta_n': 1.}
    CPT = CumulativeProspectTheory(**cpt_params)
    # print(CPT.expectation(values,p_values))
    print(CPT.expectation_PT(values,p_values))

    print(np.mean(values))
    CPT.plot_curves()

if __name__ == "__main__":
    main()
