from __future__ import division
import numpy as np
import matplotlib.pyplot as pl

""" This is code for simple GP regression. It assumes a zero mean GP Prior """
breakpoint()

# This is the true unknown function we are trying to approximate
# f = lambda x: np.sin(0.9*x).flatten()
#f = lambda x: (0.25*(x**2)).flatten()
def f(x):
    temp = np.sin(0.9*x)
    temp = temp.flatten()
    return temp

# Define the kernel
def kernel(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 0.1
    temp1 = a ** 2
    temp1 = np.sum(temp1, 1)
    temp1 = temp1.reshape(-1, 1)
    temp2 = b ** 2
    temp2 = np.sum(temp2, 1)
    temp3 = np.dot(a, b.T)
    temp3 = 2 * temp3
    temp4 = temp1 + temp2
    temp4 = temp4 - temp3
    temp4 = -0.5 * (1/kernelParameter) * temp4
    temp4 = np.exp(temp4)
    return temp4
    # sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    # return np.exp(-.5 * (1/kernelParameter) * sqdist)

N = 10         # number of training points.
n = 50         # number of test points.
s = 0.00005    # noise variance.

# Sample some input points and noisy versions of the function evaluated at
# these points.
X = np.random.uniform(-5, 5, size=(N,1))
y = f(X)
y = y + s*np.random.randn(N)

K = kernel(X, X)
L = np.linalg.cholesky(K + s*np.eye(N))

# points we're going to make predictions at.
Xtest = np.linspace(-5, 5, n).reshape(-1,1)

# compute the mean at our test points.
K_test = kernel(X, Xtest)
Lk = np.linalg.solve(L, K_test)
Ly = np.linalg.solve(L, y)
mu = np.dot(Lk.T, Ly)

# compute the variance at our test points.
K_ = kernel(Xtest, Xtest)
diag_K = np.diag(K_)
vTv = Lk**2
Lk_sum = np.sum(vTv, axis=0)
s2 = diag_K - Lk_sum
s = np.sqrt(s2)


# PLOTS:
pl.figure(1)
pl.clf()
pl.plot(X, y, 'r+', ms=20)
pl.plot(Xtest, f(Xtest), 'b-')
Xtest_flat = Xtest.flat
pl.gca().fill_between(Xtest_flat, mu-3*s, mu+3*s, color="#dddddd")
pl.plot(Xtest, mu, 'r--', lw=2)
pl.savefig('predictive.png', bbox_inches='tight')
pl.title('Mean predictions plus 3 st.deviations')
pl.axis([-5, 5, -3, 3])

# draw samples from the prior at our test points.
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
f_prior = np.dot(L, np.random.normal(size=(n,10)))
pl.figure(2)
pl.clf()
pl.plot(Xtest, f_prior)
pl.title('Ten samples from the GP prior')
pl.axis([-5, 5, -3, 3])
pl.savefig('prior.png', bbox_inches='tight')

# draw samples from the posterior at our test points.
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,10)))
pl.figure(3)
pl.clf()
pl.plot(Xtest, f_post)
pl.title('Ten samples from the GP posterior')
pl.axis([-5, 5, -3, 3])
pl.savefig('post.png', bbox_inches='tight')

pl.show()