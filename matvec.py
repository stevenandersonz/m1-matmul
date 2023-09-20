from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
import numpy as np

dim = 1024
vocab_size = 50257
x = Tensor(np.random.rand(vocab_size, dim).astype(np.float32))
linear = Linear(dim, dim, bias=True)

print(linear(x).numpy())
