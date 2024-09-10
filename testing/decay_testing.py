import numpy as np
import matplotlib.pyplot as plt

# def exponential_decay(N0, Nf, t, T):
#     w = 0.75
#     if t> T: return Nf
#     return N0 * (Nf / N0) ** ((t / T)**w)


def exponential_decay(N0, Nf, t, T, cycle=True):
    w = 0.75
    if t> T:
        # cycle through min and max decay after final iteration reached
        if cycle:
            if int(t/T) % 2 == 0: _t = t%T
            else:  _t = T - t%T
            return (N0 * (Nf / N0) ** ((_t / T)**w))
        else: return Nf
    return N0 * (Nf / N0) ** ((t / T)**w)

N0 = 0.9  # Initial value
Nf = 0.15   # Final value
T = 5000    # Total time
t = 0     # Current time
print(exponential_decay(N0, Nf, t, T))
X = [exponential_decay(N0, Nf, t, T) for t in range(3*T)]
plt.plot(X)
print([X[0],X[-1]])
plt.show()