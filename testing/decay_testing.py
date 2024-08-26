import numpy as np
import matplotlib.pyplot as plt

def exponential_decay(N0, Nf, t, T):
    w = 0.75
    if t> T: return Nf
    return N0 * (Nf / N0) ** ((t / T)**w)

N0 = 0.9  # Initial value
Nf = 0.15   # Final value
T = 5000    # Total time
t = 0     # Current time
print(exponential_decay(N0, Nf, t, T))
X = [exponential_decay(N0, Nf, t, T) for t in range(T)]
plt.plot(X)
print([X[0],X[-1]])
plt.show()