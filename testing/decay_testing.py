import numpy as np
import matplotlib.pyplot as plt

def exponential_decay(N0, Nf, t, T):
    if t> T: return Nf
    return N0 * (Nf / N0) ** (t / T)

N0 = 100  # Initial value
Nf = 10   # Final value
T = 10    # Total time
t = 10     # Current time
print(exponential_decay(N0, Nf, t, T))

plt.plot([exponential_decay(N0, Nf, t, T) for t in range(T)])
plt.show()