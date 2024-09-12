import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio
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

        self.mean_value_ref = isinstance(b, str)
        self.b = b if isinstance(b, (int, float)) else None
        self.lam = lam
        self.eta_p = eta_p
        self.eta_n = eta_n
        self.delta_p = delta_p
        self.delta_n = delta_n

    def expectation_PT(self, values, p_values):
        if  self.mean_value_ref:
            self.b = np.mean(values)

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

    def expectation(self, values, p_values):
        if self.mean_value_ref:
            self.b = np.mean(values)

        # arrange all samples in ascending order
        sorted_idxs = np.argsort(values)
        sorted_v = values[sorted_idxs]
        sorted_p = p_values[sorted_idxs]
        K = len(sorted_v)  # number of samples

        # If there is only one value, return the utility of that value
        if K==1:
            # if sorted_v[0]>self.b: return self.u_plus(sorted_v[0])
            # else: return -1*self.u_neg(sorted_v[0])
            return sorted_v[0]

        elif np.all(sorted_v<=self.b):
            Fk = [np.sum(sorted_p[0:i + 1]) for i in range(K)]
            l=K-1
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
            # Fk = [np.sum(sorted_p[0:i + 1]) for i in range(l + 1)] + \
            #      [np.sum(sorted_p[i:K]) for i in range(l + 1, K)]  # cumulative probability
            # Fk = [np.sum(sorted_p[0:i + 1]) for i in range(l)] + \
            #      [np.sum(sorted_p[i:K]) for i in range(l, K)]  # cumulative probability
            Fk = [np.sum(sorted_p[0:i + 1]) for i in range(l+1)] + \
                 [np.sum(sorted_p[i:K]) for i in range(l+1, K)]  # cumulative probability
            rho_p = self.rho_plus(sorted_v, sorted_p, Fk, l, K)
            rho_n = self.rho_neg(sorted_v, sorted_p, Fk, l, K)
            rho = rho_p - rho_n
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
        axs[0].set_xlabel('Reward')
        axs[0].set_ylabel('Perceived Reward')
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
        axs[1].set_xlabel('Probability')
        axs[1].set_ylabel('Decision Weight')
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

def main():
    animation()
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

if __name__ == "__main__":
    main()
