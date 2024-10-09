import torch
import time

t = torch.Tensor([1, 2, 3, 4, 5])

tstart = time.time()
for _ in range(100):
    torch.softmax(t, dim=0)

print(time.time()-tstart)

tstart = time.time()
for _ in range(100):
    torch.softmax(t, dim=0)

print(time.time() - tstart)