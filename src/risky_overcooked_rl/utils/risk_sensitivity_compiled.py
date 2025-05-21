import numpy as np
from numba import jit,float32
from numba.typed import List
import time
# import warnings
# warnings.simplefilter('error')
# from numba.pycc import CC
# cc = CC('risk_sensitivity_compiled')
# import matplotlib.pyplot as plt
class CumulativeProspectTheory:
    """Wrapper for compiled CPT expectations"""
    def __init__(self, b, lam, eta_p, eta_n, delta_p, delta_n):
        # self.b = np.float64(b)  # STATICALLY DEFINED REFERENCE
        # self.lam = np.float64(lam)
        # self.eta_p = np.float64(eta_p)
        # self.eta_n = np.float64(eta_n)
        # self.delta_p = np.float64(delta_p)
        # self.delta_n = np.float64(delta_n)

        self.params = {
            'b':b,
            'lam':lam,
            'eta_p':eta_p,
            'eta_n':eta_n,
            'delta_p':delta_p,
            'delta_n':delta_n
        }

    def expectation(self, values, p_values):
        return expectation_jit(values, p_values,**self.params)
    def expectation_samples(self, all_next_q_value, all_p_next_states, prospect_mask, reward, gamma):
        prospect_mask = List(prospect_mask)
        expected_td_targets = np.zeros([len(prospect_mask), 1])
        return expactation_samples_jit(expected_td_targets,
                                       all_next_q_value, all_p_next_states, prospect_mask,
                                       reward, gamma, **self.params)



@jit(nopython=True,fastmath=True)
def expactation_samples_jit(expected_td_targets,prospect_next_values, prospect_p_next_states,prospect_masks,reward,gamma,
                            b, lam, eta_p, eta_n, delta_p, delta_n):

    handle_precision_error = True
    def expectation(values, p_values, b, lam, eta_p, eta_n, delta_p, delta_n):
        """
        Calculate the expected value given values and their probabilities.
        """

        def weight(p, delta):
            # if not 0 < p <= 1: print(f"Probabilities must be between 0 and 1 v={values} p_values={p_values} p= {p}")
            return p ** delta / ((p ** delta + (1 - p) ** delta) ** (1 / delta))


        def rho_neg(sorted_v, sorted_p, Fk, l, K, b, lam, eta_n, delta_n):

            _rho_n = lam * np.abs(sorted_v[0] - b) ** eta_n \
                     * weight(sorted_p[0], delta_n)
            for i in range(1, l + 1):
                _rho_n += lam * np.abs(sorted_v[i] - b) ** eta_n \
                          * (weight(Fk[i], delta_n) - weight(Fk[i - 1], delta_n))
            return _rho_n

        def rho_plus(sorted_v, sorted_p, Fk, l, K, b, eta_p, delta_p):

            _rho_p = 0
            for i in range(l + 1, K - 1):
                _rho_p += np.abs(sorted_v[i] - b) ** eta_p * \
                          (weight(Fk[i], delta_p) - weight(Fk[i + 1], delta_p))
            # rho_p += u_plus(sorted_v[K - 1]) * w_plus(sorted_p[K - 1])
            _rho_p += np.abs(sorted_v[K - 1] - b) ** eta_p \
                      * weight(sorted_p[K - 1], delta_p)
            return _rho_p



        K = len(values)  # number of samples

        if K == 1:
            # If there is only one value, return the utility of that value
            rho_p = values[0]
            rho_n = 0
        else:
            sorted_idxs = np.argsort(values)
            sorted_v = values[sorted_idxs]
            sorted_p = p_values[sorted_idxs]  # sorted_p = p_values[sorted_idxs]
            if np.all(sorted_v <= b):
                Fk = [np.sum(sorted_p[0:i + 1]) for i in range(K)]
                if handle_precision_error: Fk = [min([max([0,f]),1]) for f in Fk]
                l = K - 1
                rho_p = 0
                rho_n = rho_neg(sorted_v, sorted_p, Fk, l, K, b, lam, eta_n, delta_n)
            elif np.all(sorted_v > b):
                Fk = [np.sum(sorted_p[i:K]) for i in range(K)]
                if handle_precision_error: Fk = [min([max([0,f]),1.0]) for f in Fk]
                l = -1
                rho_p = rho_plus(sorted_v, sorted_p, Fk, l, K, b, eta_p, delta_p)
                rho_n = 0
            else:
                l = np.where(sorted_v <= b)[0][-1]  # idx of highest loss
                Fk = [np.sum(sorted_p[0:i + 1], dtype=np.float64) for i in range(l + 1)] + \
                     [np.sum(sorted_p[i:K], dtype=np.float64) for i in range(l + 1, K)]  # cumulative probability
                if handle_precision_error: Fk = [min([max([0,f]),1]) for f in Fk]
                rho_p = rho_plus(sorted_v, sorted_p, Fk, l, K, b, eta_p, delta_p)
                rho_n = rho_neg(sorted_v, sorted_p, Fk, l, K, b, lam, eta_n, delta_n)
        rho = rho_p - rho_n
        return rho
    BATCH_SIZE = len(prospect_masks)
    # done = done.detach().cpu().numpy()
    # rewards = reward.detach().cpu().numpy()

    # expected_td_targets = np.zeros([BATCH_SIZE, 1])
    for i in range(BATCH_SIZE):
        prospect_mask = prospect_masks[i]
        prospect_values = prospect_next_values[prospect_mask, :]
        prospect_probs = prospect_p_next_states[prospect_mask, :]
        prospect_td_targets = reward[i, :] + (gamma) * prospect_values #* (1 - done[i, :]) #(solving infinite horizon)
        expected_td_targets[i] = expectation(prospect_td_targets.flatten(), prospect_probs.flatten(),
                                             b, lam, eta_p, eta_n, delta_p, delta_n)
    return expected_td_targets
# @jit(fastmath=True)
@jit
def expectation_jit(values, p_values,b, lam, eta_p, eta_n, delta_p, delta_n):
    """
    Calculate the expected value given values and their probabilities.
    """

    def weight(p, delta):
        return p ** delta / ((p ** delta + (1 - p) ** delta) ** (1 / delta))
    def rho_neg(sorted_v, sorted_p, Fk, l, K, b, lam, eta_n, delta_n):


        _rho_n = lam * np.abs(sorted_v[0] - b) ** eta_n \
                 * weight(sorted_p[0], delta_n)
        for i in range(1, l + 1):
            _rho_n += lam * np.abs(sorted_v[i] - b) ** eta_n \
                      * (weight(Fk[i], delta_n) - weight(Fk[i - 1], delta_n))
        return _rho_n

    def rho_plus(sorted_v, sorted_p, Fk, l, K, b, eta_p, delta_p):

        _rho_p = 0
        for i in range(l + 1, K - 1):
            _rho_p += np.abs(sorted_v[i] - b) ** eta_p * \
                      (weight(Fk[i], delta_p) - weight(Fk[i + 1], delta_p))
        # rho_p += u_plus(sorted_v[K - 1]) * w_plus(sorted_p[K - 1])
        _rho_p += np.abs(sorted_v[K - 1] - b) ** eta_p \
                  * weight(sorted_p[K - 1], delta_p)
        return _rho_p
    pass_single_choice = True

    sorted_idxs = np.argsort(values)
    sorted_v = values[sorted_idxs]
    sorted_p = p_values[sorted_idxs]  # sorted_p = p_values[sorted_idxs]
    K = len(sorted_v)  # number of samples

    if K == 1:
        # If there is only one value, return the utility of that value
        if pass_single_choice:
            return sorted_v[0]

    elif np.all(sorted_v <= b):
        Fk = np.clip([np.sum(sorted_p[0:i + 1]) for i in range(K)],0,1)
        l = K - 1
        rho_p = 0
        rho_n = rho_neg(sorted_v, sorted_p, Fk, l, K,b,lam,eta_n,delta_n)
        rho = rho_p - rho_n
        return rho
    elif np.all(sorted_v > b):
        Fk = np.clip([np.min(np.sum(sorted_p[i:K]),1) for i in range(K)],0,1)
        l = -1
        rho_p = rho_plus(sorted_v, sorted_p, Fk, l, K,b,eta_p,delta_p)
        rho_n = 0
        rho = rho_p - rho_n
        return rho
    else:
        l = np.where(sorted_v <= b)[0][-1]  # idx of highest loss
        Fk = np.clip([np.sum(sorted_p[0:i + 1], dtype=np.float64) for i in range(l + 1)] + \
             [np.sum(sorted_p[i:K], dtype=np.float64) for i in range(l + 1, K)],0,1)  # cumulative probability
        rho_p = rho_plus(sorted_v, sorted_p, Fk, l, K,b,eta_p,delta_p)
        rho_n = rho_neg(sorted_v, sorted_p, Fk, l, K,b,lam,eta_n,delta_n)
        rho = rho_p - rho_n
        return rho



def main():
    # Example usage
    values = np.array([0.1, 0.2, 0.3])
    p_values = np.array([0.5, 0.3, 0.2])
    cpt_params = {
        'b': 0.5,
        'lam': 2,
        'eta_p': 1,
        'eta_n': 1,
        'delta_p': 1,
        'delta_n': 1
    }
    # calc time of loop
    start = time.time()
    for _ in range(1000000):
         result = expectation_jit(values, p_values, **cpt_params)
    end = time.time()
    print(f"Standard Compiling: Time taken: {end - start} seconds")
if __name__ == "__main__":
    main()
