import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
breakpoint()

# def f_x(x):
#     return -(x**3)+(6*(x**2))-x+17
#
# result = integrate.quad(f_x, -2, 5)
# print(result)
#
# def Monte_Carlo_estimator(f, low, high, N=1000):
#     x = np.random.uniform(low, high, size=N)
#     total = 0
#     width = high - low
#     for i in x:
#         area = width * f(i)
#         total += area
#     return total/N
#
#
# print(Monte_Carlo_estimator(f_x, -2, 5, N=10000))

N = 10000
a, b = (50,50)
x_min, x_max = (0, .55)
randx = np.random.uniform(x_min, x_max, N)
y = stats.beta.pdf(randx, a, b)
print(f'Real value to find: {stats.beta(a,b).cdf(.55)}')
print(f'Integral value:  {(x_max-x_min)*y.sum()/N}')
print(f'Calculation error: {np.sqrt((x_max-x_min)*(y*y).sum()/N - (x_max-x_min)*y.mean()**2)/np.sqrt(N)}')

# Plotting
plt.figure(figsize=(8,5))
x = np.linspace(0, 1, 1000)
plt.plot(x, stats.beta.pdf(x, a, b))
# Then, let's only plot a thousand points for more readability
plt.scatter(randx[:1000], y[:1000], alpha=.08, label='value of f for random uniform x')
plt.xlabel('x')
plt.ylabel('density')
plt.legend()
plt.show()


N = 10000
a, b = (50,50)
x_min, x_max = (0, .55)

randx = np.random.uniform(x_min,x_max, N)
y = stats.beta.pdf(randx, a, b)
randy = np.random.uniform(0,y.max(), N)
print(f'Integral value: {(x_max-x_min)*y.max()*(randy <= y).sum()/N}')

plt.figure(figsize=(8,5))
color = randy[:1000] <= y[:1000]
x = np.linspace(0, 1, 1000)
plt.plot(x, stats.beta.pdf(x, a, b))
plt.scatter(randx[:1000], randy[:1000], alpha=.2, c = color)
plt.xlabel('x')
plt.ylabel('density')
plt.show()


n = 10000
a, b = (50,50)
x_min, x_max = (0, .55)

mean, var = stats.beta.stats(a, b, moments='mv')

randx = np.random.normal(mean, 1.25*np.sqrt(var), size=n)
randx_uniform = np.random.uniform(x_min, x_max, size=n)

randx = randx[(randx >= x_min) & (randx <= x_max)]

f = stats.beta(a,b).pdf(randx)
f_uniform = stats.beta(a,b).pdf(randx_uniform)
g = stats.norm.pdf(randx, mean, 1.25*np.sqrt(var))

y = f / g

print(f'Integral value: {1*y.sum()/n}')
print(f'Error: {np.sqrt(1/n * (y * y).sum() - (1/n * y.sum())**2)/np.sqrt(n)}', end = '\n\n')

fig, axs = plt.subplots(1, 2, figsize=(15,5))

axs[0].scatter(randx[:1000], f[:1000], alpha=.06, label='f')
axs[1].scatter(randx_uniform[:1000], f_uniform[:1000], alpha=.06, label='f_uniform')
axs[0].scatter(randx[:1000], g[:1000], alpha=.01, label='g')
axs[0].scatter(randx[:1000], y[:1000], alpha=.005, label='f / g')

x = np.linspace(0, 1, 1000)
x_to_fill = np.array([0 for _ in range(len(x))])
x_to_fill[(x>=x_min) & (x<=x_max)] = 1

axs[0].plot(x, stats.beta.pdf(x, a, b), alpha= .3)
axs[1].plot(x, stats.beta.pdf(x, a, b), alpha= .3)
axs[0].fill_between(x, stats.beta.pdf(x, a, b), alpha=.02, where=x_to_fill, hatch= '/')
axs[1].fill_between(x, stats.beta.pdf(x, a, b), alpha=.02, where=x_to_fill, hatch= '/')

axs[0].set_xlabel('x')
axs[0].set_ylabel('density')
axs[1].set_xlabel('x')
axs[1].set_ylabel('density')
axs[0].legend()
axs[1].legend()
axs[0].set_xlim(0,1)
axs[1].set_xlim(0,1)
plt.show()












