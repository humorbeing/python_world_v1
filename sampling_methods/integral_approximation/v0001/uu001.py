

import numpy as np


x = np.random.uniform(size=1000)

print(x.mean())

x = np.random.normal(size=1000)
y = np.zeros_like(x)


for i, xx in enumerate(x):
    if xx >= 1.0:
        y[i] = 1

print(y.mean())