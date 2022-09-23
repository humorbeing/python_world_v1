import numpy as np
import matplotlib.pyplot as plt
breakpoint()
def kernel(a, b):
    """GP squared exponential kernel"""
    temp1 = a**2
    temp1 = np.sum(temp1, 1)
    temp1 = temp1.reshape(-1, 1)
    temp2 = b**2
    temp2 = np.sum(temp2, 1)
    temp3 = np.dot(a, b.T)
    temp3 = 2*temp3
    temp4 = temp1 + temp2
    temp4 = temp4 - temp3
    temp4 = np.exp(-0.5*temp4)
    # sqdist = np.sum(a**2, 1).reshape(-1,1)\
    # + np.sum(b**2, 1) - 2*np.dot(a, b.T)
    return temp4

n = 50
Xtest = np.linspace(-5, 5, n)
Xtest = Xtest.reshape(-1, 1)
K_ = kernel(Xtest, Xtest)

L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
sameple_y = np.random.normal(size=(n, 2))
f_prior = np.dot(L, sameple_y)

plt.plot(Xtest, f_prior)

plt.show()





