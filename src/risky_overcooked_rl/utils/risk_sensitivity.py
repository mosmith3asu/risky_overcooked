import itertools
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import torch
from numba import int32, float32,boolean    # import the types
from numba.experimental import jitclass
from tqdm import trange,tqdm
import torch
from torch.cuda.amp import autocast



spec = [
    ('b', float32),           # a simple scalar field
    ('lam', float32),
    ('eta_p', float32),
    ('eta_n', float32),
    ('delta_p', float32),
    ('delta_n', float32),
    ('mean_value_ref', boolean),
    ('is_rational', boolean),
    ('expected_td_targets', float32[:])
]

@jitclass(spec)
class CumulativeProspectTheory_Compiled:
    def __init__(self,b,lam,eta_p,eta_n,delta_p,delta_n,mean_value_ref = False):
        """
        Instantiates a CPT object that can be used to model human risk-sensitivity.
        :param b: reference point determining if outcome is gain or loss
        :param lam: loss-aversion parameter
        :param eta_p: exponential gain on positive outcomes
        :param eta_n: exponential loss on negative outcomes
        :param delta_p: probability weighting for positive outcomes
        :param delta_n: probability weighting for negative outcomes
        """
        # assert b==0, "Reference point must be 0"
        self.b = b
        self.lam = lam
        self.eta_p = eta_p
        self.eta_n = eta_n
        self.delta_p = delta_p
        self.delta_n = delta_n

        self.mean_value_ref = mean_value_ref
        if self.mean_value_ref:
            self.b = 0

        self.expected_td_targets = np.zeros(256, dtype=np.float32)

        self.is_rational = True
        if self.lam !=1: self.is_rational = False
        elif self.eta_p != 1: self.is_rational = False
        elif self.eta_n != 1: self.is_rational = False
        elif self.delta_p != 1: self.is_rational = False
        elif self.delta_n != 1: self.is_rational = False

    def expectation_samples(self,prospect_next_values, prospect_p_next_states,prospect_masks,reward,gamma,done):

        BATCH_SIZE = len(prospect_masks)
        self.expected_td_targets = np.zeros(BATCH_SIZE, dtype=np.float32)
        for i in range(BATCH_SIZE):
            prospect_mask = prospect_masks[i]
            prospect_values = prospect_next_values[prospect_mask, :]
            prospect_probs = prospect_p_next_states[prospect_mask, :]
            prospect_td_targets = reward[i, :] + (gamma) * prospect_values# * (1 - done[i, :]) #(solving infinite horizon)

            if self.is_rational:
                self.expected_td_targets[i] = np.sum(prospect_td_targets.flatten() * prospect_probs.flatten())
            else:
                self.expected_td_targets[i] = self.expectation(prospect_td_targets.flatten(), prospect_probs.flatten())


        return self.expected_td_targets


    def _get_l(self,sorted_v):
        K = len(sorted_v)  # number of samples
        if K == 1: return sorted_v[0]  # Single prospect = no CPT
        l = np.where(sorted_v <= self.b)[0]
        l = -1 if len(l) == 0 else l[-1]  # of no losses l=-1 indicator
        return K,l
    def _get_Fk(self,sorted_p,K,l):
        # Step 2: Calculate the cumulative liklihoods for gains and losses
        Fk = [min([max([0, np.sum(sorted_p[0:i + 1])]), 1]) for i in range(l + 1)] + \
             [min([max([0, np.sum(sorted_p[i:K])]), 1]) for i in range(l + 1, K)]  # cumulative probability
        Fk = Fk + [0]  # padding to make dealing with only gains or only losses easier
        return Fk
    def expectation(self,values, p_values):
        """
        Applies the CPT-expectation multiple prospects (i.e. a series of value-probability pairs) which can arbitrarily
        replace the rational expectation operator E[v,p] = Σ(p*v). When dealing with more than two prospects, we must
        calculate the expectation over the cumulative probability distributions.
        :param values:
        :param p_values:
        :return:
        """

        values = values.astype(np.float64)
        p_values = p_values.astype(np.float64)
        if self.is_rational:
            # Rational Expectation
            return np.sum(values * p_values)
        if self.mean_value_ref:
            self.b = np.mean(values)

        # Step 1: arrange all samples in ascending order and get indexs of gains/losses
        sorted_idxs = np.argsort(values)
        sorted_v = values[sorted_idxs]
        sorted_p = p_values[sorted_idxs]

        K = len(sorted_v)  # number of samples
        if K == 1: return sorted_v[0]  # Single prospect = no CPT
        l = np.where(sorted_v <= self.b)[0]
        l = -1 if len(l) == 0 else l[-1]  # of no losses l=-1 indicator

        # Step 2: Calculate the cumulative liklihoods for gains and losses
        Fk = [min([max([0, np.sum(sorted_p[0:i + 1])]), 1]) for i in range(l + 1)] + \
             [min([max([0, np.sum(sorted_p[i:K])]), 1]) for i in range(l + 1, K)]  # cumulative probability
        Fk = Fk + [0]  # padding to make dealing with only gains or only losses easier

        # Step 3: Calculate biased expectation for gains and losses
        rho_p = self.perc_util_plus(sorted_v, Fk, l, K)
        rho_n = self.perc_util_neg(sorted_v, Fk, l, K)

        # Step 3: Add the cumulative expectation and return
        rho = rho_p - rho_n

        if self.mean_value_ref:
            rho += self.b
            self.b = 0
        return rho

    def perc_util_plus(self,sorted_v,Fk,l,K):
        """Calculates the cumulative expectation of all utilities percieved as gains"""
        rho_p = 0
        for i in range(l + 1, K):
            rho_p += self.u_plus(sorted_v[i]) * (self.w_plus(Fk[i]) - self.w_plus(Fk[i + 1]))
        # CLASSICAL FORMULATION ( no Fk =  Fk + [0]) -----------------------
        # for i in range(l + 1, K - 1):
        #     rho_p += self.u_plus(sorted_v[i]) * (self.w_plus(Fk[i]) - self.w_plus(Fk[i + 1]))
        # rho_p += self.u_plus(sorted_v[K - 1]) * self.w_plus(sorted_p[K - 1])
        return rho_p

    def perc_util_neg(self,sorted_v,Fk,l,K):
        """Calculates the cumulative expectation of all utilities percieved as losses"""
        # Fk =  Fk + [0]  # add buffer which results in commented out version below
        rho_n = 0
        for i in range(0, l + 1):
            rho_n += self.u_neg(sorted_v[i]) * (self.w_neg(Fk[i]) - self.w_neg(Fk[i - 1]))
        return rho_n
        # CLASSICAL FORMULATION ( no Fk =  Fk + [0]) -----------------------
        # rho_n = self.u_neg(sorted_v[0]) * self.w_neg(sorted_p[0])
        # for i in range(1, l + 1):
        #     rho_n += self.u_neg(sorted_v[i]) * (self.w_neg(Fk[i]) - self.w_neg(Fk[i - 1]))
        # return rho_n

    def u_plus(self,v):
        """ Weights the values (v) perceived as losses (v>b)"""
        return np.abs(v-self.b)**self.eta_p
    def u_neg(self, v):
        """ Weights the values (v) perceived as gains (v<=b)"""
        return self.lam * np.abs(v-self.b) ** self.eta_n
    def w_plus(self, p):
        """ Weights the probabilities p for probabilities of values perceived as gains  (v>b)"""
        delta = self.delta_p
        return p ** delta / ((p ** delta + (1 - p) ** delta) ** (1 / delta))
    def w_neg(self, p):
        """ Weights the probabilities p for probabilities of values perceived as losses (v<=b)"""
        delta = self.delta_n
        return p ** delta / ((p ** delta + (1 - p) ** delta) ** (1 / delta))


class CumulativeProspectTheory:
    def __init__(self, b, lam, eta_p, eta_n, delta_p, delta_n, compiled=False,mean_value_ref=False):
        """
        Instantiates a CPT object that can be used to model human risk-sensitivity.
        :param b: reference point determining if outcome is gain or loss
        :param lam: loss-aversion parameter
        :param eta_p: exponential gain on positive outcomes
        :param eta_n: exponential loss on negative outcomes
        :param delta_p: probability weighting for positive outcomes
        :param delta_n: probability weighting for negative outcomes
        """
        # assert b==0, "Reference point must be 0"
        self.b = b
        self.lam = lam
        self.eta_p = eta_p
        self.eta_n = eta_n
        self.delta_p = delta_p
        self.delta_n = delta_n

        self.compiled_expectation = compiled
        # if self.compiled_expectation:
        #     self.expectation_batch = torch.jit.script(self.expectation_batch)

        self._f_vmap = None

        self.is_rational = True
        if self.b != 0:
            self.is_rational = False
        elif self.lam != 1:
            self.is_rational = False
        elif self.eta_p != 1:
            self.is_rational = False
        elif self.eta_n != 1:
            self.is_rational = False
        elif self.delta_p != 1:
            self.is_rational = False
        elif self.delta_n != 1:
            self.is_rational = False

    def sample_expectation_batch(self, X):
        """Estimates CPT-value from only samples (no probs) in batch form"""
        raise NotImplementedError("sample_expectation_batch not implemented yet.")

    def sample_expectation(self, X):
        raise NotImplementedError("sample_expectation not implemented yet.")

    def expectation(self, values, p_values):
        """
        Applies the CPT-expectation multiple prospects (i.e. a series of value-probability pairs) which can arbitrarily
        replace the rational expectation operator E[v,p] = Σ(p*v). When dealing with more than two prospects, we must
        calculate the expectation over the cumulative probability distributions.
        :param values:
        :param p_values:
        :return:
        """
        if isinstance(values, np.ndarray):
            values = torch.tensor(values, dtype=torch.float64).reshape(1,-1)
        if isinstance(p_values, np.ndarray):
            p_values = torch.tensor(p_values, dtype=torch.float64).reshape(1,-1)
        return self.expectation_batch(values, p_values)

        # if self.is_rational:
        #     # Rational Expectation
        #     return np.sum(values * p_values)
        #
        # # Step 1: arrange all samples in ascending order and get indexs of gains/losses
        # sorted_idxs = np.argsort(values)
        # sorted_v = values[sorted_idxs]
        # sorted_p = p_values[sorted_idxs]
        #
        # K = len(sorted_v)  # number of samples
        # if K == 1: return sorted_v[0]  # Single prospect = no CPT
        # l = np.where(sorted_v <= self.b)[0]
        # l = -1 if len(l) == 0 else l[-1]  # of no losses l=-1 indicator
        #
        # # Step 2: Calculate the cumulative liklihoods for gains and losses
        # Fk = [min([max([0, np.sum(sorted_p[0:i + 1])]), 1]) for i in range(l + 1)] + \
        #      [min([max([0, np.sum(sorted_p[i:K])]), 1]) for i in range(l + 1, K)]  # cumulative probability
        # Fk = Fk + [0]  # padding to make dealing with only gains or only losses easier
        #
        # # Step 3: Calculate biased expectation for gains and losses
        # rho_p = self.perc_util_plus(sorted_v, Fk, l, K)
        # rho_n = self.perc_util_neg(sorted_v, Fk, l, K)
        #
        # # Step 3: Add the cumulative expectation and return
        # rho = rho_p - rho_n
        #
        # return rho


        # if add_pad:
        #     from itertools import zip_longest
        #     X = list(zip(*zip_longest(*X, fillvalue=0)))
        #     p_values = list(zip(*zip_longest(*p_values, fillvalue=0)))
        #
        #     X = torch.tensor(X, device=device, dtype=torch.float64)
        #     p_values = torch.tensor(p_values,device=device, dtype=torch.float64)

    def expectation_batch(self, X: torch.Tensor, p: torch.Tensor, add_pad = False) -> torch.Tensor:
        """
        Calculates CPT expectation in batches from prospects of size (B, N)
        where B is batch size and N is number of prospects.
        """
        if add_pad:
            from itertools import zip_longest
            device = X[0].device
            X = list(zip(*zip_longest(*X, fillvalue=torch.inf)))
            p_values = list(zip(*zip_longest(*p, fillvalue=0)))
            X = torch.tensor(X, device=device, dtype=torch.float64)
            p = torch.tensor(p_values,device=device, dtype=torch.float64)

        assert isinstance(X, torch.Tensor)
        assert isinstance(p, torch.Tensor)

        B, N = X.shape
        device = X.device
        _dtype = X.dtype

        dtype = torch.float64
        X.to(dtype=torch.float64)
        p.to(dtype=torch.float64)

        if self.compiled_expectation:
            pt_params = [torch.tensor(param, dtype=torch.float32, device=device)
                         for param in [self.b, self.lam, self.eta_p, self.eta_n, self.delta_p, self.delta_n]]
            rho = _compiled_expectation(X, p, *pt_params)
            return rho

        # Ensure contiguous (helps some kernels, especially after slicing/gathering)
        X = X.contiguous()
        p = p.contiguous()

        # ---- Sort values and align probabilities ----
        X_sort, sorted_idxs = torch.sort(X, dim=1)  # [B, N]
        P_sort = torch.gather(p.to(device=device, dtype=dtype), 1, sorted_idxs)

        # Index of last loss for each batch (<= b)
        L = (X_sort <= self.b).sum(dim=1) - 1  # [B]

        # Precompute (or reuse from self if N fixed) index row: [1, N] broadcasts to [B, N]
        idx = torch.arange(N, device=device).view(1, -1)

        # mask_minus: losses; mask_plus: gains
        mask_minus = idx <= L.unsqueeze(1)  # [B, N]
        mask_plus = ~mask_minus  # [B, N]
        # Note: when all gains, L = -1 → mask_minus all False, mask_plus all True
        #       when all losses, L = N-1 → mask_minus all True, mask_plus all False

        # ---- Cumulative probabilities ----
        # Forward cumsum for losses
        F_minus = P_sort.cumsum(dim=1)  # [B, N]


        # Reverse cumsum for gains
        P_rev = torch.flip(P_sort, dims=[1])
        F_plus = torch.flip(P_rev.cumsum(dim=1), dims=[1])  # [B, N]

        # Select correct cumulative probs per entry without extra masks
        F_minus = F_minus.clamp(min=0.0, max=1.0) # Handle rounding errors
        F_plus = F_plus.clamp(min=0.0, max=1.0) # Handle rounding errors
        # assert torch.all(F_plus >= 0) and torch.all(F_plus <= 1), f"F_plus out of bounds {F_plus}"
        # assert torch.all(F_minus >= 0) and torch.all(F_minus <= 1), f"F_minus out of bounds {F_minus}"
        Fk = torch.where(mask_minus, F_minus, F_plus)
        # Fk = torch.where(mask_minus, F_minus, F_plus)  # [B, N]

        # ---- Shifted Fk for decision weights ----
        pad0 = X.new_zeros(B, 1)  # correct device & dtype

        # Keep the same structure as your original implementation
        z1 = torch.cat([Fk, pad0], dim=1)[:, :-1]  # [B, N]
        z2 = torch.cat([Fk, pad0], dim=1)[:, 1:]  # [B, N]
        z3 = torch.cat([pad0, Fk], dim=1)[:, 1:]  # [B, N]
        z4 = torch.cat([pad0, Fk], dim=1)[:, :-1]  # [B, N]
        _b = torch.tensor(self.b, device=X.device, dtype=X.dtype)

        # ---- Utility and weighting ----
        u_plus = self.u_plus(X_sort)
        u_minus = self.u_neg(X_sort)

        rho_plus    = u_plus * (self.w_plus(z1) - self.w_plus(z2))
        rho_minus   = u_minus * (self.w_neg(z3) - self.w_neg(z4))

        # Only gains contribute to rho_plus, only losses to rho_minus
        rho_plus =  torch.nan_to_num(rho_plus,  nan=0.0, posinf=0.0) * mask_plus
        rho_minus = torch.nan_to_num(rho_minus, nan=0.0, posinf=0.0) * mask_minus

        # Final CPT expectation per batch element
        rho = rho_plus.sum(dim=1) - rho_minus.sum(dim=1)  # [B]

        return rho.to(dtype=_dtype)

    def u_plus(self, v):
        """ Weights the values (v) perceived as losses (v>b)"""
        if isinstance(v,torch.Tensor):
            return torch.pow(torch.abs(v - self.b), self.eta_p)
        return np.power(np.abs(v - self.b), self.eta_p)

    def u_neg(self, v):
        """ Weights the values (v) perceived as gains (v<=b)"""
        if isinstance(v,torch.Tensor):
            return self.lam * torch.pow(torch.abs(v - self.b), self.eta_n)
        return self.lam * np.power(np.abs(v - self.b), self.eta_n)

    def w_plus(self, p):
        """ Weights the probabilities p for probabilities of values perceived as gains  (v>b)"""
        return self._w(p, self.delta_p)

    def w_neg(self, p):
        """ Weights the probabilities p for probabilities of values perceived as losses (v<=b)"""
        return self._w(p, self.delta_n)

    def _w(self, p, delta):
        if isinstance(p,torch.Tensor):
            z = torch.pow(p, delta)  # precompute term
            denom = z + torch.pow(1 - p, delta)
            denom = torch.pow(denom, 1 / delta)
            return z / denom

        z = np.power(p,delta)  # precompute term
        denom = z + np.power(1 - p,delta)
        denom = np.power(denom, 1 / delta)
        return z / denom
        # return p ** delta / ((p ** delta + (1 - p) ** delta) ** (1 / delta))




@torch.jit.script
def _utility_fun(v,b,lam,eta_n):
    return lam * torch.pow(torch.abs(v - b), eta_n)

@torch.jit.script
def _weight_fun(p,delta):
    z = torch.pow(p, delta)  # precompute term
    denom = z + torch.pow(1 - p, delta)
    denom = torch.pow(denom, 1 / delta)
    return z / denom

@torch.jit.script
def _compiled_expectation(X, p, b, lam_n, eta_p, eta_n, delta_p, delta_n):
    B, N = X.shape
    device = X.device
    dtype = X.dtype

    # Ensure contiguous (helps some kernels, especially after slicing/gathering)
    X = X.contiguous()
    p = p.contiguous()

    # ---- Sort values and align probabilities ----
    X_sort, sorted_idxs = torch.sort(X, dim=1)  # [B, N]
    P_sort = torch.gather(p.to(device=device, dtype=dtype), 1, sorted_idxs)

    # Index of last loss for each batch (<= b)
    L = (X_sort <= b).sum(dim=1) - 1  # [B]

    # Precompute (or reuse from self if N fixed) index row: [1, N] broadcasts to [B, N]
    idx = torch.arange(N, device=device).view(1, -1)

    # mask_minus: losses; mask_plus: gains
    mask_minus = idx <= L.unsqueeze(1)  # [B, N]
    mask_plus = ~mask_minus  # [B, N]
    # Note: when all gains, L = -1 → mask_minus all False, mask_plus all True
    #       when all losses, L = N-1 → mask_minus all True, mask_plus all False

    # ---- Cumulative probabilities ----
    # Forward cumsum for losses
    F_minus = P_sort.cumsum(dim=1)  # [B, N]

    # Reverse cumsum for gains
    P_rev = torch.flip(P_sort, dims=[1])
    F_plus = torch.flip(P_rev.cumsum(dim=1), dims=[1])  # [B, N]

    # Select correct cumulative probs per entry without extra masks
    F_minus = F_minus.clamp(min=0.0, max=1.0)  # Handle rounding errors
    F_plus = F_plus.clamp(min=0.0, max=1.0)  # Handle rounding errors
    Fk = torch.where(mask_minus, F_minus, F_plus)  # [B, N]


    # ---- Shifted Fk for decision weights ----
    pad0 = X.new_zeros(B, 1)  # correct device & dtype

    # Keep the same structure as your original implementation
    z1 = torch.cat([Fk, pad0], dim=1)[:, :-1]  # [B, N]
    z2 = torch.cat([Fk, pad0], dim=1)[:, 1:]  # [B, N]
    z3 = torch.cat([pad0, Fk], dim=1)[:, 1:]  # [B, N]
    z4 = torch.cat([pad0, Fk], dim=1)[:, :-1]  # [B, N]
    # return X_sort, P_sort, L, mask_plus, mask_minus, z1, z2, z3, z4

    # ---- Utility and weighting ----
    lam_p = torch.ones_like(lam_n)
    u_plus = _utility_fun(X_sort, b,lam_p,eta_p)
    u_minus = _utility_fun(X_sort, b, lam_n, eta_n)

    rho_plus = u_plus * (_weight_fun(z1,delta_p) - _weight_fun(z2,delta_p))
    rho_minus = u_minus * (_weight_fun(z3,delta_n) - _weight_fun(z4,delta_n))

    # Only gains contribute to rho_plus, only losses to rho_minus
    rho_plus = torch.nan_to_num(rho_plus, nan=0.0, posinf=0.0) * mask_plus
    rho_minus = torch.nan_to_num(rho_minus, nan=0.0, posinf=0.0) * mask_minus
    # rho_plus = rho_plus * mask_plus
    # rho_minus = rho_minus * mask_minus

    # Final CPT expectation per batch element
    # rho = torch.nan_to_num(rho_plus - rho_minus).sum(dim=1)  # [B]
    rho = rho_plus.sum(dim=1) - rho_minus.sum(dim=1)  # [B]
    return rho

def gen_random_prospect(min_val=-10, max_val=10, min_size=2, max_size=10):
    size = np.random.randint(min_size, max_size + 1)
    # values = np.random.randint(min_val, max_val + 1, size)
    values = np.random.rand(size)* (max_val - min_val) + min_val
    p_values = np.random.rand(size)
    p_values /= p_values.sum()
    return values, p_values


def test_equivalance(cpt1, cpt2, n_tests=1000):
    # CASE: centered/zero expectation
    values = np.array([-20, -10, -5.0, 5.0, 10, 20])
    p_values = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
    _values = np.array([-5, -5, 5, 5])
    _p_values = np.array([0.3, 0.2, .2, 0.3])
    vbatch = [torch.tensor(values.copy()), torch.tensor(_values.copy())]
    pbatch = [torch.tensor(p_values.copy()), torch.tensor(_p_values.copy())]

    expectation = [cpt1.expectation(values.copy(), p_values.copy()),
                   cpt1.expectation(_values.copy(), _p_values.copy())]
    expectations_new = cpt2.expectation_batch(vbatch, pbatch, add_pad=True)
    assert np.allclose(expectation,
                       expectations_new.cpu().numpy()), f'[CASE MIXED] Values do not match: {expectation} vs {expectations_new.cpu().numpy()}'

    # Mixed
    for _ in range(100):
        values, p_values = gen_random_prospect()
        # _values, _p_values = gen_random_prospect()
        _values, _p_values = values, p_values
        vbatch = [torch.tensor(values), torch.tensor(_values)]
        pbatch = [torch.tensor(p_values), torch.tensor(_p_values)]

        expectation = [cpt1.expectation(values.copy(), p_values.copy()),
                       cpt1.expectation(_values.copy(), _p_values.copy())]
        expectations_new = cpt2.expectation_batch(vbatch, pbatch, add_pad=True)
        assert np.allclose(expectation,
                           expectations_new.cpu().numpy()), f'[CASE MIXED] Values do not match: {expectation} vs {expectations_new.cpu().numpy()}'

    # All negatives
    for _ in range(100):
        values, p_values = gen_random_prospect(min_val=-5, max_val=-1)
        # _values, _p_values = gen_random_prospect(min_val=-5, max_val=-1)
        _values, _p_values = values, p_values
        vbatch = [torch.tensor(values), torch.tensor(_values)]
        pbatch = [torch.tensor(p_values), torch.tensor(_p_values)]

        expectation = [cpt1.expectation(values.copy(), p_values.copy()),
                       cpt1.expectation(_values.copy(), _p_values.copy())]

        expectations_new = cpt2.expectation_batch(vbatch, pbatch, add_pad=True)
        assert np.allclose(expectation,
                           expectations_new.cpu().numpy()), f'[CASE NEGATIVE] Values do not match: {expectation} vs {expectations_new.cpu().numpy()}'

    # All positive
    for _ in range(100):
        values, p_values = gen_random_prospect(min_val=1, max_val=20)
        _values, _p_values = gen_random_prospect(min_val=1, max_val=20)
        vbatch = [torch.tensor(values), torch.tensor(_values)]
        pbatch = [torch.tensor(p_values), torch.tensor(_p_values)]

        expectation = [cpt1.expectation(values.copy(), p_values.copy()),
                       cpt1.expectation(_values.copy(), _p_values.copy())]

        expectations_new = cpt2.expectation_batch(vbatch, pbatch, add_pad=True)
        assert np.allclose(expectation,
                           expectations_new.cpu().numpy()), f'Values do not match: {expectation} vs {expectations_new.cpu().numpy()}'

def test_dominance():
    import tqdm
    rat_params = {
        'b': 0.0,
        'lam':1.0,
        'eta_p': 1.0,
        'eta_n': 1.0,
        'delta_p': 1.0,
        'delta_n':1.0,
        'mean_value_ref': False
    }
    seek_params = {
        'b': 0.0,
        'lam':0.44,
        'eta_p': 1.0,
        'eta_n': 0.88,
        'delta_p': 1.0,
        'delta_n':1.0,
        'mean_value_ref': True
    }
    averse_params = {
        'b': 0.0,
        'lam':2.25,
        'eta_p': 0.88,
        'eta_n': 1.0,
        'delta_p': 1.0,
        'delta_n':1.0,
        'mean_value_ref': True
    }


    # averse = CumulativeProspectTheory_Compiled(**averse_params)
    cpt_rat = CumulativeProspectTheory_Compiled(**rat_params)
    cpt_seek = CumulativeProspectTheory_Compiled(**seek_params)
    cpt_aver = CumulativeProspectTheory_Compiled(**averse_params)
    n_tests = 100000

    for _ in tqdm.tqdm(range(n_tests)):
        values, p_values = gen_random_prospect()
        val_rat = cpt_rat.expectation(values.copy(), p_values.copy())
        val_seek = cpt_seek.expectation(values.copy(), p_values.copy())
        # val_aver = cpt_aver.expectation(values.copy(), p_values.copy())
        # print(np.array([ val_aver,val_rat, val_seek]).round(2))

        assert val_seek >= val_rat, f"Seeking not dominating rational: {val_seek} vs {val_rat} \nValues: {values} with probs {p_values}"
        # assert val_aver <= val_rat, f"Averse not dominated by rational: {val_aver} vs {val_rat}"

def test_indifference():
    neutral_params = {
        'b': 0.0,
        'lam':1.0,
        'eta_p': 1.0,
        'eta_n': 1.0,
        'delta_p': 1.0,
        'delta_n':1.0,
        # 'mean_value_ref': True
    }
    averse_params = {
        'b': 0.0,
        'lam': 2.25,
        'eta_p': 0.88,
        'eta_n': 1.0,
        # 'delta_p': 1,
        # 'delta_n': 1,
        'delta_p': 0.61,
        'delta_n': 0.69,
        # 'mean_value_ref': True
    }
    seeking_params = {
        'b': 0.0,
        'lam': 0.44,
        'eta_p': 1,
        'eta_n':  0.88,

        # 'delta_p': 1,
        # 'delta_n': 1,
        'delta_p': 0.61,
        'delta_n': 0.69,
    }


    cpt1 = CumulativeProspectTheory_Compiled(**neutral_params)
    cpt2 = CumulativeProspectTheory(compiled=True, **seeking_params)
    cpt3 = CumulativeProspectTheory(compiled=True, **averse_params)
    fig, axs = plt.subplots(3,3,figsize=(6, 6), constrained_layout=True)




    ######################################
    x_vals = (200, 100, 0)
    ax_row = axs[0]
    indifference_curve(ax_row[0], cpt2, n_points=100, x_vals=x_vals)
    indifference_curve(ax_row[1], cpt1, n_points=100, x_vals=x_vals)
    indifference_curve(ax_row[2], cpt3, n_points=100, x_vals=x_vals, colorbar=True)
    for ax in ax_row:
        # annotate xvals in top right of plot
        ax.annotate(f'Nonnegative:: {x_vals}', xy=(0.95, 0.95), xycoords='axes fraction',
                    fontsize=8, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=0.5))

   #######################################

    x_vals = (-200, 0, 200)
    print(f"Plotting indifference curves... {x_vals}")
    ax_row = axs[1]
    indifference_curve(ax_row[0], cpt2, n_points=100, x_vals=x_vals)
    indifference_curve(ax_row[1], cpt1, n_points=100,x_vals=x_vals)
    indifference_curve(ax_row[2], cpt3, n_points=100,x_vals=x_vals, colorbar=True)
    ax_row[0].set_title('Seeking')
    ax_row[1].set_title('Rational')
    ax_row[2].set_title('Averse')
    ax_row[0].set_ylabel('p3')
    for ax in ax_row:
        # annotate xvals in top right of plot
        ax.annotate(f'Mixed: {x_vals}', xy=(0.95, 0.95), xycoords='axes fraction',
                    fontsize=8, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=0.5))



    ######################################
    x_vals = (-200, -100, 0)
    ax_row = axs[2]
    indifference_curve(ax_row[0], cpt2, n_points=100, x_vals=x_vals, colorbar=True)
    indifference_curve(ax_row[1], cpt1, n_points=100, x_vals=x_vals, colorbar=True)
    indifference_curve(ax_row[2], cpt3, n_points=100, x_vals=x_vals, colorbar=True)
    for ax in ax_row:
        # annotate xvals in top right of plot
        ax.annotate(f'Nonpositive: {x_vals}', xy=(0.95, 0.95), xycoords='axes fraction',
                    fontsize=8, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=0.5))
    plt.show()


def indifference_curve(ax, cpt, x_vals=(-200, -100, 0), n_points=100,colorbar=False):
    P1 = np.linspace(0, 1, n_points)
    P3 = np.linspace(0, 1, n_points)

    Xg, Yg = np.meshgrid(P1, P3)
    Zg = np.zeros_like(Xg)

    for i in trange(n_points):
        for j in range(n_points):
            p1 = P1[i]
            p3 = P3[j]
            p2 = 1 - p1 - p3
            if any(0 > p < 1 for p in [p1, p2, p3]):
                Zg[i, j] = None
            else:
                values = np.array(x_vals)
                p_values = np.array([p1, p2,p3])
                cpt_val = cpt.expectation(values, p_values)
                if isinstance(cpt_val, torch.Tensor):
                    cpt_val = cpt_val.detach().cpu().numpy()
                # Zg[i, j] = np.round(cpt_val - np.sum(x_vals * p_values),2)
                Zg[i, j] =cpt_val

                # print(f'val{cpt.expectation(values, p_values)}')


    # plot meshgrid as 3d surface
    # surf = ax.plot_surface(Xg, Yg, Zg, #cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)

    # ax.pcolormesh(Xg, Yg, Zg, shading='auto', cmap='RdBu')
    cb = ax.contourf(Xg, Yg, Zg)
    if colorbar:
        plt.colorbar(cb, ax=ax, label='intensity')


    # contour = ax.contour(Xg, Yg, Zg, levels=[0], colors='blue')
    # ax.clabel(contour, inline=True, fontsize=8)


    # ax.set_xlabel('p1')
    # ax.set_ylabel('p2')
    # ax.set_title('Indifference Curve (CPT)')

    # add white triangle patch
    # triangle = plt.Polygon([[1, 1], [1, 0], [0, 1]], color='white', zorder=10)
    # ax.add_patch(triangle)



def main():
    test_indifference()
    # test_dominance()
    # neutral_params = {
    #     'b': 0.0,
    #     'lam':1.0,
    #     'eta_p': 1.0,
    #     'eta_n': 1.0,
    #     'delta_p': 1.0,
    #     'delta_n':1.0,
    #     # 'mean_value_ref': True
    # }
    # averse_params = {
    #     'b': 0.0,
    #     'lam': 2.25,
    #     'eta_p': 0.88,
    #     'eta_n': 1.0,
    #     'delta_p': 0.61,
    #     'delta_n': 0.69,
    #     # 'mean_value_ref': True
    # }
    # seeking_params = {
    #     'b': 0.0,
    #     'lam': 0.44,
    #     'eta_p': 1,
    #     'eta_n':  0.88,
    #     'delta_p': 0.61,
    #     'delta_n': 0.69,
    #     # 'mean_value_ref': True
    # }

    # test1_params = {
    #     'b': 0.0,
    #     'lam': 2.25,
    #     'eta_p': 0.88,
    #     'eta_n': 1.0,
    #     'delta_p': 0.61,
    #     'delta_n': 0.69,
    #     # 'mean_value_ref': True
    # }
    #
    #
    # # averse = CumulativeProspectTheory_Compiled(**averse_params)
    # cpt = CumulativeProspectTheory_Compiled(**test1_params)
    # cpt_new = CumulativeProspectTheory(compiled=True, **test1_params)
    # test_equivalance(cpt, cpt_new, n_tests=1000)
    #
    # test_speed()




if __name__ == "__main__":
    main()


