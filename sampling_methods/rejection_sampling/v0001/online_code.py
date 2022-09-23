import numpy as np
import scipy.stats as st
# import seaborn as sns
import matplotlib.pyplot as plt

breakpoint()
# sns.set()
# proposal_distribution_normal = True
proposal_distribution_normal = False

def p(x):
    return st.norm.pdf(x, loc=30, scale=10) + st.norm.pdf(x, loc=80, scale=20)


def q(x):
    if proposal_distribution_normal:
        return st.norm.pdf(x, loc=50, scale=30)
        # return st.uniform.pdf(x, loc=-50, scale=200)
    else:
        return np.ones_like(x)

x = np.arange(-50, 151)
k = max(p(x) / q(x))


def rejection_sampling(iter=1000):
    samples = []

    for i in range(iter):
        if proposal_distribution_normal:
            z = np.random.normal(50, 30)
        else:
            z = np.random.uniform(-60, 160)
        u = np.random.uniform(0, k*q(z))

        if u <= p(z):
            samples.append(z)

    return np.array(samples)


if __name__ == '__main__':
    plt.plot(x, p(x))
    plt.plot(x, k*q(x))
    plt.show()

    s = rejection_sampling(iter=100000)
    # sns.distplot(s)
    # sns.displot(s)
    plt.hist(s, 50, density=True, facecolor='g', alpha=0.75)
    plt.show()









