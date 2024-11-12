import itertools
import time

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio
import random



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
        assert b==0, "Reference point must be 0"
        self.mean_value_ref = False
        self.exp_value_ref = False
        self.exp_rational_value_ref = False
        if isinstance(b, str):
            assert b in ['m','e','erat'], "b must be either ['m':mean,'e':expected,'erat':expected rational]"
            if b == 'm': self.mean_value_ref = True
            elif b == 'e': self.exp_value_ref = True
            elif b == 'erat': self.exp_rational_value_ref = True
            else: raise NotImplementedError()
            self.b = None
        else: self.b = np.float64(b) # STATICALLY DEFINED REFERENCE
        self.lam = np.float64(lam)
        self.eta_p = np.float64(eta_p)
        self.eta_n = np.float64(eta_n)
        self.delta_p = np.float64(delta_p)
        self.delta_n = np.float64(delta_n)
        # self.lam = lam
        # self.eta_p = eta_p
        # self.eta_n = eta_n
        # self.delta_p = delta_p
        # self.delta_n = delta_n
        self.N_samples =100
    def expectation_PT(self, values, p_values, value_refs=None):
        """
        Computes Prospect Theory Expectation (for validation)
        - 2 values only
             :param values: list of values of next states
        :param p_values: probability of each value
        :param value_refs: list of (rational) values used to compute reference point
        :return: scalar value (biased) expectation
        """
        if self.mean_value_ref: self.b = np.mean(values)
        elif self.exp_value_ref:  self.b = np.sum(values * p_values)
        elif self.exp_rational_value_ref:  self.b = np.sum(value_refs * p_values)

        # arrange all samples in ascending order
        vp = values[np.where(values > self.b)[0]]
        vn = values[np.where(values <= self.b)[0]]
        u_plus = self.u_plus(vp)
        u_neg = -1 * self.u_neg(vn)
        u = np.hstack([u_neg, u_plus])


        pp = p_values[np.where(values > self.b)[0]]
        pn = p_values[np.where(values <= self.b)[0]]
        w_plus = self.w_plus(pp)
        w_neg = self.w_neg(pn)
        w = np.hstack([w_neg, w_plus])
        rho = np.sum(u*w)
        return rho

    def expectation(self, values, p_values, value_refs=None, pass_single_choice=True):
        """
        Computes CUMULATIVE Prospect Theory Expectation
        :param values: list of values of next states
        :param p_values: probability of each value
        :param value_refs: list of (rational) values used to compute reference point
        :param pass_single_choice: if True, return the rational value of the single prospect
        :return: scalar value (biased) expectation
        """
        # arrange all samples in ascending order
        sorted_idxs = np.argsort(values)
        sorted_v = values[sorted_idxs]
        sorted_p = p_values[sorted_idxs]
        K = len(sorted_v)  # number of samples

        if K == 1:
            # If there is only one value, return the utility of that value
            if pass_single_choice: return sorted_v[0]
            elif sorted_v[0] > self.b:return self.u_plus(sorted_v[0])
            else: return -1 * self.u_neg(sorted_v[0])

        elif np.all(sorted_v <= self.b):
            Fk = [np.sum(sorted_p[0:i + 1]) for i in range(K)]
            l = K - 1
            rho_p = 0
            rho_n = self.rho_neg(sorted_v, sorted_p, Fk, l, K)
            rho = rho_p - rho_n
            return rho
        elif np.all(sorted_v > self.b):
            Fk = [np.sum(sorted_p[i:K]) for i in range(K)]
            l = -1
            rho_p = self.rho_plus(sorted_v, sorted_p, Fk, l, K)
            rho_n = 0
            rho = rho_p - rho_n
            return rho
        else:
            l = np.where(sorted_v <= self.b)[0][-1]  # idx of highest loss
            Fk = [np.sum(sorted_p[0:i + 1], dtype=np.float64) for i in range(l + 1)] + \
                 [np.sum(sorted_p[i:K], dtype=np.float64) for i in range(l + 1, K)]  # cumulative probability
            rho_p = self.rho_plus(sorted_v, sorted_p, Fk, l, K)
            rho_n = self.rho_neg(sorted_v, sorted_p, Fk, l, K)
            rho = rho_p - rho_n
            return rho
    # def expectation(self, values, p_values, value_refs=None):
    #     """
    #     Computes CUMULATIVE Prospect Theory Expectation
    #     :param values: list of values of next states
    #     :param p_values: probability of each value
    #     :param value_refs: list of (rational) values used to compute reference point
    #     :return: scalar value (biased) expectation
    #     """
    #     if self.mean_value_ref: self.b = np.mean(values)
    #     elif self.exp_value_ref: self.b = np.sum(values*p_values)
    #     elif self.exp_rational_value_ref: self.b = np.sum(value_refs*p_values)
    #     # arrange all samples in ascending order
    #     sorted_idxs = np.argsort(values)
    #     sorted_v = values[sorted_idxs]
    #     sorted_p = p_values[sorted_idxs]
    #     K = len(sorted_v)  # number of samples
    #
    #     # If there is only one value, return the utility of that value
    #     if K==1:
    #         # if sorted_v[0]>self.b: return self.u_plus(sorted_v[0])
    #         # else: return -1*self.u_neg(sorted_v[0])
    #         return sorted_v[0]
    #
    #     elif np.all(sorted_v<=self.b):
    #         Fk = [np.sum(sorted_p[0:i + 1]) for i in range(K)]
    #         l=K-1
    #         rho_p = 0
    #         rho_n = self.rho_neg(sorted_v, sorted_p, Fk, l, K)
    #         rho = rho_p - rho_n
    #         return rho
    #     elif np.all(sorted_v > self.b):
    #         Fk = [np.sum(sorted_p[i:K]) for i in range(K)]
    #         l = -1
    #         rho_p = self.rho_plus(sorted_v, sorted_p, Fk, l, K)
    #         rho_n = 0
    #         rho = rho_p - rho_n
    #         return rho
    #     else:
    #         l = np.where(sorted_v <= self.b)[0][-1]  # idx of highest loss
    #         Fk = [np.sum(sorted_p[0:i + 1],dtype=np.float64) for i in range(l+1)] + \
    #              [np.sum(sorted_p[i:K],dtype=np.float64) for i in range(l+1, K)]  # cumulative probability
    #         rho_p = self.rho_plus(sorted_v, sorted_p, Fk, l, K)
    #         rho_n = self.rho_neg(sorted_v, sorted_p, Fk, l, K)
    #         rho = rho_p - rho_n
    #         return rho

    def expectation3(self,values,p_values):
        sorted_idxs = np.argsort(values)
        X_sort = values[sorted_idxs]
        P_sort = p_values[sorted_idxs]
        K = len(X_sort)

        # number of losses
        l = np.where(X_sort <= self.b)[0]  # idx of highest loss
        if len(l) == 0: l = -1  # no losses
        else: l = l[-1]

        F = np.nan * np.ones(X_sort.shape)
        for k in range(K):
            if k <= l: F[k] = np.sum(P_sort[0:k + 1])
            else:  F[k] = np.sum(P_sort[k:K + 1])
        assert np.all(np.isfinite(F)), "F is not finite"

        rho_plus = self.util_plus(X_sort[K-1],self.eta_p) * self.weight(P_sort[K-1],self.delta_p)
        rho_minus = self.util_minus(X_sort[0], self.lam, self.eta_n) * self.weight(P_sort[0], self.delta_n)
        for i in range(1,K-1):
            rho_plus += self.util_plus(X_sort[i],self.eta_p) * (self.weight(F[i],self.delta_p) - self.weight(F[i + 1],self.delta_p))
            rho_minus += self.util_minus(X_sort[i], self.lam, self.eta_n) * (self.weight(F[i], self.delta_n) - self.weight(F[i -1], self.delta_n))
        rho = rho_plus - rho_minus
        return rho

    def expectation2(self,values,p_values,pass_single_choice=False):
        # sorted_idxs = np.argsort(values)
        # X_sort = values[sorted_idxs]
        # P_sort = p_values[sorted_idxs]
        # K = len(X_sort)-1  # number of samples
        # l = np.where(X_sort <= self.b)[0][-1]  # idx of highest loss
        #
        # F = np.nan*np.ones(X_sort.shape)
        # for k in range(K+1):
        #     if k<=l: F[k] = np.sum(P_sort[0:k+1])
        #     else: F[k] = np.sum(P_sort[k:K+1])
        # assert np.all(np.isfinite(F)), "F is not finite"
        #
        # rho_plus = 0
        # for i in range(l+1, K):
        #     rho_plus += self.u_plus(X_sort[i]) * (self.w_plus(F[i]) - self.w_plus(F[i+1]))
        # rho_plus += self.u_plus(X_sort[K]) * self.w_plus(P_sort[K])
        #
        # rho_minus = self.u_neg(X_sort[0]) * self.w_neg(P_sort[0])
        # for i in range(1, l+1):
        #     rho_minus += self.u_neg(X_sort[i]) * (self.w_neg(F[i]) - self.w_neg(F[i-1]))
        #
        # rho = rho_plus - rho_minus
        # return rho
        sorted_idxs = np.argsort(values)
        X_sort = values[sorted_idxs]
        P_sort = p_values[sorted_idxs]
        K = len(X_sort)

        # number of losses
        l = np.where(X_sort <= self.b)[0]  # idx of highest loss
        if len(l)==0: l=-1 # no losses
        else: l = l[-1] +1

        F = np.nan * np.ones(X_sort.shape)
        for k in range(K):
            if k <= l: F[k] = np.sum(P_sort[0:k + 1])
            else: F[k] = np.sum(P_sort[k:K + 1])
        assert np.all(np.isfinite(F)), "F is not finite"

        rho_plus = 0
        if l<K:
            for i in range(l + 1, K-1):
                rho_plus += self.u_plus(X_sort[i]) * (self.w_plus(F[i]) - self.w_plus(F[i + 1]))
            rho_plus += self.u_plus(X_sort[K]) * self.w_plus(P_sort[K])


        rho_minus = self.u_neg(X_sort[0]) * self.w_neg(P_sort[0])
        for i in range(1, l + 1):
            rho_minus += self.u_neg(X_sort[i]) * (self.w_neg(F[i]) - self.w_neg(F[i - 1]))

        rho = rho_plus - rho_minus
        return rho
    def expectation_from_samples(self,values,p_values):
        N_max = self.N_samples
        X = [np.random.choice(values,p=p_values)  for _ in range(N_max)] # generate empirical samples
        X_sort = np.sort(X)

        rho_plus, rho_minus = 0, 0
        _rho_plus, _rho_minus = 0, 0
        for i in range(1, N_max + 1):
            z_1 = (N_max + 1 - i) / N_max # # {z_1 = (N_max + i - 1) / N_max <-- mistake}
            z_2 = (N_max - i) / N_max
            z_3 = i / N_max
            z_4 = (i - 1) / N_max

            _rho_plus += self.util_plus(X_sort[i - 1], self.eta_p) * (
                        self.weight(z_1, self.delta_p) - self.weight(z_2, self.delta_p))

            _rho_minus += self.util_minus(X_sort[i - 1], self.lam,self.eta_n) * (
                        self.weight(z_3, self.delta_n) - self.weight(z_4, self.delta_n))

        rho = rho_plus - rho_minus
        return rho

    def rho_plus(self,sorted_v,sorted_p,Fk,l,K):
        rho_p = 0
        for i in range(l + 1, K - 1):
            rho_p += self.u_plus(sorted_v[i]) * (self.w_plus(Fk[i]) - self.w_plus(Fk[i + 1]))
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

    def util_plus(self,rew,eta):
        rew = max(rew, 0)
        return np.power(abs(rew), eta)

    def util_minus(self,rew,lam,beta):
        rew = min(rew, 0)
        return lam * np.power(abs(rew), beta)

    def weight(self,prob, gamma):
        # assert 0<=prob<=1, "prob must be in [0,1]"
        return (prob ** gamma) / (((prob ** gamma) + (1 - prob) ** gamma) ** (1 / gamma))

    def plot_curves(self,with_params=False,get_img=False,neg_lambda=True):
        # c_gain = 'tab:green'
        # c_loss = 'tab:red'
        c_gain = 'royalblue'
        c_loss = 'firebrick'
        if with_params:
            fig, axs = plt.subplots(1, 3, constrained_layout=True,figsize=(9,3))
        else:
            fig,axs = plt.subplots(1,2,constrained_layout=True,figsize=(10,5))

        # Plot utility functions
        v = np.linspace(-10,10,100)
        vp = v[np.where(v>=self.b)[0]]
        vn = v[np.where(v<=self.b)[0]]
        u_plus = self.u_plus(vp)
        u_neg = -1*self.u_neg(vn)
        u = np.hstack([u_neg,u_plus])
        axs[0].plot(vp,u_plus,color=c_gain,label='$u^+(r)$')
        axs[0].plot(np.hstack([vn,vp[0]]),np.hstack([u_neg,u_plus[0]]),color=c_loss,label='$u^-(r)$')
        # axs[0].plot(v, u, color='g',label='Rational')
        axs[0].plot(v, v, color='gray',linestyle='--',label='Rational')
        axs[0].set_xlabel('Value $\\tau_i$')
        axs[0].set_ylabel('Perceived Value $u(\\tau_i)$')
        axs[0].set_ylim([min(v),max(v)])
        axs[0].plot([v[0], v[-1]], [0, 0], c="lightgrey", zorder=1, lw=1)
        axs[0].plot([0, 0], [v[0], v[-1]], c="lightgrey", zorder=1, lw=1)
        #make axis square
        axs[0].set_aspect('equal', adjustable='box')
        axs[0].legend(frameon=False,ncol=1)
        # axs[0].set_xlim([0,10])
        # axs[0].set_ylim([0, 10])
        # Plot probability weighting functions
        p = np.linspace(0,1,100)
        pp = self.w_plus(p)
        pn = self.w_neg(p)
        axs[1].plot(p, pp,color=c_gain,label='$w^+(p)$')
        axs[1].plot(p, pn,color=c_loss,label='$w^-(p)$')
        axs[1].plot(p, p, color='gray', label='Rational',linestyle='--')
        axs[1].set_xlabel('Probability $p_i$')
        axs[1].set_ylabel('Decision Weight $w(p_i)$')
        axs[1].legend(frameon=False,ncol=1)
        axs[1].set_aspect('equal', adjustable='box')
        axs[1].set_ylim([0, 1])
        axs[1].set_xlim([0, 1])
        if with_params:
            if neg_lambda:
                param_dict = {
                    '$b$': self.b,
                    '$1/\ell$': 1/self.lam,
                    '$\eta^+$': self.eta_p,
                    '$\eta^-$': self.eta_n,
                    '$\delta^+$': self.delta_p,
                    '$\delta^-$': self.delta_n
                }
            else:
                param_dict = {
                     '$b$':self.b,
                 '$\ell$':self.lam,
                 '$\eta^+$':self.eta_p,
                 '$\eta^-$':self.eta_n,
                 '$\delta^+$':self.delta_p,
                 '$\delta^-$':self.delta_n
                }

            # create a barchart of param_dict
            c = (255/255, 192/255, 0)
            axs[2].bar(param_dict.keys(),param_dict.values(), fc = c)
            # axs[2].bar(param_dict.keys(), param_dict.values(), fc=['k','r','b','r','b','r'])
            for bar,c in zip(axs[2].get_children(),['tab:gray',c_loss,c_gain,c_loss,c_gain,c_loss]):
                if isinstance(bar, plt.Rectangle):
                    bar.set_color(c)
            axs[2].set_aspect(1.0/axs[2].get_data_ratio(), adjustable='box')
            axs[2].set_ylim([0,1.1])

            axs[2].set_ylabel('Parameter Value')
        if get_img:
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            return np.asarray(buf)

        else:
            plt.show()

    @property
    def is_rational(self):
        return (self.b==0 and self.lam==1
                and self.eta_p == 1 and self.eta_n == 1
                and self.delta_p == 1 and self.delta_n == 1)
def animation():
    fps = 10
    fname = 'CPT_animation'
    IMGS = []
    cpt_params = {'b': 0, 'lam': 1.0,
                  'eta_p': 1., 'eta_n': 1.,
                  'delta_p': 1., 'delta_n': 1.}

    N = 30
    # for eta_p,eta_n,delta_p,delta_n in zip(
    #                         np.linspace(1, 0.4, N),
    #                        np.linspace(1, 0.1, N),
    #                        np.linspace(1, 0.5, N),
    #                        np.linspace(1, 0.2, N)):
    #     cpt_params['eta_p'] = eta_p
    #     cpt_params['eta_n'] = eta_n
    #     cpt_params['delta_p'] = delta_p
    #     cpt_params['delta_n'] = delta_n
    #
    #     CPT = CumulativeProspectTheory(**cpt_params)
    #     IMGS.append(CPT.plot_curves(with_params=True,get_img=True))

    # for lam in 1/np.linspace(1, 0.5, N):
    #     cpt_params['lam'] = lam
    #     CPT = CumulativeProspectTheory(**cpt_params)
    #     IMGS.append(CPT.plot_curves(with_params=True, get_img=True,neg_lambda=True))
    # for lam in 1/np.linspace(0.5, 1, N):
    #     cpt_params['lam'] = lam
    #     CPT = CumulativeProspectTheory(**cpt_params)
    #     IMGS.append(CPT.plot_curves(with_params=True, get_img=True,neg_lambda=True))

    # for b in np.linspace(0, 1, N):
    #     cpt_params['b'] = b
    #     CPT = CumulativeProspectTheory(**cpt_params)
    #     IMGS.append(CPT.plot_curves(with_params=True,get_img=True))
    # for b in np.linspace(1, 0, N):
    #     cpt_params['b'] = b
    #     CPT = CumulativeProspectTheory(**cpt_params)
    #     IMGS.append(CPT.plot_curves(with_params=True, get_img=True))

    # for eta_p,eta_n in zip(
    #                         np.linspace(1, 0.3, N),
    #                        np.linspace(1, 0.7, N)):
    #     cpt_params['eta_p'] = eta_p
    #     cpt_params['eta_n'] = eta_n
    #     # cpt_params['delta_p'] = delta_p
    #     # cpt_params['delta_n'] = delta_n
    #
    #     CPT = CumulativeProspectTheory(**cpt_params)
    #     IMGS.append(CPT.plot_curves(with_params=True,get_img=True))
    #
    # for eta_p, eta_n in zip(
    #         np.linspace(0.3,1, N),
    #         np.linspace(0.7,1, N)):
    #     cpt_params['eta_p'] = eta_p
    #     cpt_params['eta_n'] = eta_n
    #     # cpt_params['delta_p'] = delta_p
    #     # cpt_params['delta_n'] = delta_n
    #
    #     CPT = CumulativeProspectTheory(**cpt_params)
    #     IMGS.append(CPT.plot_curves(with_params=True,get_img=True))

    # for delta_p,delta_n in zip(
    #                         np.linspace(1, 0.6, N),
    #                        np.linspace(1, 0.4, N)):
    #     cpt_params['delta_p'] = delta_p
    #     cpt_params['delta_n'] = delta_n
    #     CPT = CumulativeProspectTheory(**cpt_params)
    #     IMGS.append(CPT.plot_curves(with_params=True,get_img=True))
    # for delta_p, delta_n in zip(
    #         np.linspace(0.6,1, N),
    #         np.linspace(0.4,1, N)):
    #     cpt_params['delta_p'] = delta_p
    #     cpt_params['delta_n'] = delta_n
    #     CPT = CumulativeProspectTheory(**cpt_params)
    #     IMGS.append(CPT.plot_curves(with_params=True, get_img=True))
    # fname = 'CPT_animation_delta'

    for lam,eta_p,eta_n,delta_p,delta_n in zip(
            1 / np.linspace(1, 0.5, N),
            np.linspace(1, 0.3, N),
            np.linspace(1, 0.7, N),
                            np.linspace(1, 0.6, N),
                           np.linspace(1, 0.4, N)):

        cpt_params['eta_p'] = eta_p
        cpt_params['eta_n'] = eta_n
        cpt_params['delta_p'] = delta_p
        cpt_params['delta_n'] = delta_n
        # cpt_params['lam'] = lam
        CPT = CumulativeProspectTheory(**cpt_params)
        IMGS.append(CPT.plot_curves(with_params=True, get_img=True,neg_lambda=True))
    for lam, eta_p, eta_n, delta_p, delta_n in reversed(list(zip(
            1 / np.linspace(1, 0.5, N),
            np.linspace(1, 0.3, N),
            np.linspace(1, 0.7, N),
            np.linspace(1, 0.6, N),
            np.linspace(1, 0.4, N)))):
        cpt_params['eta_p'] = eta_p
        cpt_params['eta_n'] = eta_n
        cpt_params['delta_p'] = delta_p
        cpt_params['delta_n'] = delta_n
        # cpt_params['lam'] = lam
        CPT = CumulativeProspectTheory(**cpt_params)
        IMGS.append(CPT.plot_curves(with_params=True, get_img=True, neg_lambda=True))
    # for delta_p, delta_n in zip(
    #         np.linspace(0.6,1, N),
    #         np.linspace(0.4,1, N)):
    #     cpt_params['delta_p'] = delta_p
    #     cpt_params['delta_n'] = delta_n
    #     CPT = CumulativeProspectTheory(**cpt_params)
    #     IMGS.append(CPT.plot_curves(with_params=True, get_img=True))
    fname = 'CPT_animation_all'
    print(len(IMGS))
    imageio.mimsave(f'{fname}.gif', IMGS,loop=0,fps=fps)
def animation_all():
    fps = 10
    fname = 'CPT_animation'
    IMGS = []
    cpt_params = {'b': 0, 'lam': 1.0,
                  'eta_p': 1., 'eta_n': 1.,
                  'delta_p': 1., 'delta_n': 1.}



    N = 30
    sp_b = np.linspace(0, 1, N)
    sp_lam = 1/np.linspace(1, 0.5, N)
    sp_eta_p = np.linspace(1, 0.3, N)
    sp_eta_n = np.linspace(1, 0.8, N)
    sp_delta_p = np.linspace(1, 0.6, N)
    sp_delta_n = np.linspace(1, 0.4, N)

    for lam,eta_p,eta_n,delta_p,delta_n in zip(
            sp_lam,sp_eta_p,sp_eta_n,sp_delta_p,sp_delta_n):
        cpt_params['eta_p'] = eta_p
        cpt_params['eta_n'] = eta_n
        cpt_params['delta_p'] = delta_p
        cpt_params['delta_n'] = delta_n
        # cpt_params['lam'] = lam
        CPT = CumulativeProspectTheory(**cpt_params)
        IMGS.append(CPT.plot_curves(with_params=True, get_img=True,neg_lambda=True))
    for lam, eta_p, eta_n, delta_p, delta_n in reversed(list(zip(
            sp_lam, sp_eta_p, sp_eta_n, sp_delta_p, sp_delta_n))):
        cpt_params['eta_p'] = eta_p
        cpt_params['eta_n'] = eta_n
        cpt_params['delta_p'] = delta_p
        cpt_params['delta_n'] = delta_n
        # cpt_params['lam'] = lam
        CPT = CumulativeProspectTheory(**cpt_params)
        IMGS.append(CPT.plot_curves(with_params=True, get_img=True, neg_lambda=True))
    fname = 'CPT_animation_all'
    print(len(IMGS))
    imageio.mimsave(f'{fname}.gif', IMGS,loop=0,fps=fps)

def main():
    animation_all()
    # animation()
    # # example from https://engineering.purdue.edu/DELP/education/decision_making_slides/Module_12___Cumulative_Prospect_Theory.pdf
    # # values = np.array([80,60,40,20,0])
    # # p_values = np.array([0.2,0.2,0.2,0.2,0.2])
    # values = np.array([-10, -20, -1,1, 10, 20])
    # # values = np.array([-80, 60, 40, 20, 0])
    # p_values = np.array([0.2, 0.2, 0.2,0.2, 0.2, 0.2])
    #
    # # values =  np.array([-0.018, -0.018])#np.random.randint(-5,5,2)
    # values = np.array([0.02, 0.02])  # np.random.randint(-5,5,2)
    # p_values = np.array([0.5, 0.5])
    #
    # values = np.array([-0.00110984, -0.00128507])  # np.random.randint(-5,5,2)
    # p_values = np.array([0.1, 0.9])
    #
    # # values = np.array([-1,1])
    # # p_values = np.ones(len(values)) / len(values)
    # # cpt_params = {'b':2.0, 'lam':2.0,
    # #               'eta_p':0.88,'eta_n':0.5,
    # #               'delta_p':0.88,'delta_n':0.6}
    # cpt_params = {'b': 0, 'lam': 1.0,
    #               'eta_p': 1., 'eta_n': 1.,
    #               'delta_p': 1., 'delta_n': 1.}
    # # cpt_params = {'b':1.0, 'lam':2.25,
    # #               'eta_p':0.88,'eta_n':0.5,
    # #               'delta_p':0.88,'delta_n':0.6}
    # CPT = CumulativeProspectTheory(**cpt_params)
    # print(CPT.expectation(values,p_values))
    # print(CPT.expectation_PT(values,p_values))
    #
    # print(np.sum(values*p_values))
    # CPT.plot_curves()
    cpt_params = {'b': 0, 'lam': 2.25,
                  'eta_p': 0.88, 'eta_n': 0.88,
                  'delta_p': 0.61, 'delta_n': 0.69}
    CPT = CumulativeProspectTheory(**cpt_params)
    CPT.plot_curves()
#########################################################################################
#########################################################################################
    # values = np.array([0.01073038, 0.01114906])
    # values = np.array([-15,-10,-5.0, 5.0, 10, 15])
    # p_values = np.array([0.1,0.2,0.2, 0.2,0.2, 0.1])
    # values = np.array([-15,-10,-5.0])*-1
    # p_values = np.array([0.2,0.3,0.5])
    values = np.array([-15])
    p_values = np.array([1])

    # FROM Stochastic systems with cumulative prospect theory
    # Example 2:
    # values = np.array([-5, -3, -1, 2, 4, 6])
    # p_values = np.array([1/6,1/6,1/6,1/6,1/6,1/6])#np.ones_like(values)/len(values)


    # cpt_params = {'b': 0, 'lam': 1,
    #               'eta_p':0.88,'eta_n':0.88,
    #               'delta_p':0.61,'delta_n':"0.69"}
    # cpt_params = {'b': 'e', 'lam': 1,
    #               'eta_p':1,'eta_n':1,
    #               'delta_p':0.61,'delta_n':0.69}
    cpt_params = {'b': 0, 'lam': 1,
                  'eta_p':0.88,'eta_n':0.88,
                  'delta_p':0.61,'delta_n':0.69}
    # cpt_params = {'b': 0, 'lam': 1,
    #               'eta_p':1,'eta_n':0.88,
    #               'delta_p':1,'delta_n':1}
    CPT = CumulativeProspectTheory(**cpt_params)
    for p_slip in [0.1,0.25,0.5,0.75,0.9]:
        print(f'p_slip = {p_slip}: w^-= {round(CPT.w_plus(1-p_slip),3)} | w^-= {round(CPT.w_neg(p_slip),3)}')
    # print(CPT.expectation_from_samples(values, p_values))

    # print(CPT.expectation(values, p_values))
    # print(CPT.expectation_PT(values, p_values))

    # tstart = time.time()
    # for _ in range(10000):
    #     v_exp = CPT.expectation3(values, p_values)
    # print(f'Exp1: {v_exp} {time.time()-tstart} ')

    tstart = time.time()
    for _ in range(10000):
        v_exp = CPT.expectation(values, p_values)
    print(f'Exp1: {v_exp} {time.time()-tstart} ')

    # tstart = time.time()
    # for _ in range(10000):
    #     v_exp = CPT.expectation2(values, p_values)
    # print(f'Exp2: {v_exp} {time.time() - tstart} ')


    print(np.sum(values * p_values))
if __name__ == "__main__":
    main()

    b = 0
    eta_p = [0,1,2]
    eta_n = [0, 1, 2]
    delta_p = [0, 1, 2]
    delta_n = [0, 1, 2]
    params = itertools.product(*[eta_p,eta_n,delta_p,delta_n])
    print(len(list(params)))