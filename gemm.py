import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import numpy as np 
import time


dim = 1024
A = np.random.rand(dim, dim).astype(np.float32)
B = np.random.rand(dim, dim).astype(np.float32)

st = time.monotonic()
C = A @ B
et = time.monotonic()
s = (et - st)
flop = (dim * dim * dim * 2) * 1e-9
print(f"numpy's {(flop/s):.2f} gflop/s" )

for nm, M in [('A', A),('B',B),('C',C)]:
    with open(f"{nm}.txt", "w") as f:
        for row in M:
            line = ' '.join(map(str, row))
            f.write(f"{line}\n")
