# import numpy as np
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from corner import corner

breakpoint()
def mh_sampler(x0, lnprob_fn, prop_fn, prop_fn_kwargs={}, iterations=100000):
    """Simple metropolis hastings sampler.

    :param x0: Initial array of parameters.
    :param lnprob_fn: Function to compute log-posterior.
    :param prop_fn: Function to perform jumps.
    :param prop_fn_kwargs: Keyword arguments for proposal function
    :param iterations: Number of iterations to run sampler. Default=100000

    :returns:
        (chain, acceptance, lnprob) tuple of parameter chain , acceptance rate
        and log-posterior chain.
    """

    # number of dimensions
    ndim = len(x0)

    # initialize chain, acceptance rate and lnprob
    chain = np.zeros((iterations, ndim))
    lnprob = np.zeros(iterations)
    accept_rate = np.zeros(iterations)

    # first samples
    chain[0] = x0
    lnprob0 = lnprob_fn(x0)
    lnprob[0] = lnprob0

    # start loop
    naccept = 0
    for ii in range(1, iterations):

        # propose
        x_star, factor = prop_fn(x0, **prop_fn_kwargs)

        # draw random uniform number
        u = np.random.uniform(0, 1)

        # compute hastings ratio
        lnprob_star = lnprob_fn(x_star)
        H = np.exp(lnprob_star - lnprob0) * factor

        # accept/reject step (update acceptance counter)
        if u < H:
            x0 = x_star
            lnprob0 = lnprob_star
            naccept += 1

        # update chain
        chain[ii] = x0
        lnprob[ii] = lnprob0
        accept_rate[ii] = naccept / ii

    return chain, accept_rate, lnprob


def gaussian_proposal(x, sigma=0.1):
    """
    Gaussian proposal distribution.

    Draw new parameters from Gaussian distribution with
    mean at current position and standard deviation sigma.

    Since the mean is the current position and the standard
    deviation is fixed. This proposal is symmetric so the ratio
    of proposal densities is 1.

    :param x: Parameter array
    :param sigma:
        Standard deviation of Gaussian distribution. Can be scalar
        or vector of length(x)

    :returns: (new parameters, ratio of proposal densities)
    """

    # Draw x_star
    x_star = x + np.random.randn(len(x)) * sigma

    # proposal ratio factor is 1 since jump is symmetric
    qxx = 1

    return (x_star, qxx)


def simple_gaussian_lnpost(x):
    """
    1-D Gaussian distribution with mean 0 std 1.

    Prior on mean is U(-10, 10)

    :param x: Array of parameters

    :returns: log-posterior

    """
    mu = 0
    std = 1

    if x < 10 and x > -10:
        return -0.5 * (x - mu) ** 2 / std ** 2
    else:
        return -1e6


# xx = np.linspace(-3.5, 3.5, 1000)
# x0 = np.array([-1.6])
# xs = np.array([-1.0])
# fig = plt.figure(figsize=(8,5))
# ax = plt.subplot(111)
#
# ax.plot(xx, scipy.stats.norm(loc=0, scale=1).pdf(xx)/scipy.stats.norm(loc=0, scale=1).pdf(xx).max(),
#         label='Posterior Distribution', lw=2)
# ax.plot(xx, scipy.stats.norm(loc=x0, scale=0.3).pdf(xx)/scipy.stats.norm(loc=x0, scale=0.3).pdf(xx).max(),
#         alpha=1.0, label='Proposal Distribution', lw=2)
# ax.plot(xx, scipy.stats.norm(loc=xs, scale=0.3).pdf(xx)/scipy.stats.norm(loc=x0, scale=0.3).pdf(xx).max(),
#         alpha=0.2, color='k', lw=2)
# ax.plot(x0, scipy.stats.norm(loc=0, scale=1).pdf(x0)/scipy.stats.norm(loc=0, scale=1).pdf(xx).max(), 'o',
#         label='Initial', markersize=10)
# ax.plot(xs, scipy.stats.norm(loc=0, scale=1).pdf(xs)/scipy.stats.norm(loc=0, scale=1).pdf(xx).max(), 'o',
#         label='Proposed', markersize=10)
# ax.axvline(xs, ls='--', color='k', alpha=0.3)
# ax.axvline(x0, ls='--', color='C1', alpha=0.3)
#
# # posterior ratio
# pr = np.exp(simple_gaussian_lnpost(xs) - simple_gaussian_lnpost(x0))
#
# # proposal ratio
# qxy = scipy.stats.norm(loc=xs, scale=0.3).pdf(x0)
# qyx = scipy.stats.norm(loc=x0, scale=0.3).pdf(xs)
# H = pr * qxy/qyx
#
# ax.set_title('$\pi(x_*)/\pi(x_0)$ = {:2.2}\n $q(x_0|x_*)/q(x_*|x_0)$ = {:2.2}\n $H$ = {:2.2}'.format(pr[0], (qxy/qyx)[0], H[0]),
#              fontsize=14)
#
# ax.set_ylim(ymin=0)
# plt.legend(loc='best', frameon=False)
# ax.set_xlabel(r'$x$', fontsize=14);
# plt.show()
#
# xx = np.linspace(-3.5, 3.5, 1000)
# x0 = np.array([-1.6])
# xs = np.array([-2.2])
# fig = plt.figure(figsize=(8,5))
# ax = plt.subplot(111)
#
# ax.plot(xx, scipy.stats.norm(loc=0, scale=1).pdf(xx)/scipy.stats.norm(loc=0, scale=1).pdf(xx).max(),
#         label='Posterior Distribution', lw=2)
# ax.plot(xx, scipy.stats.norm(loc=x0, scale=0.3).pdf(xx)/scipy.stats.norm(loc=x0, scale=0.3).pdf(xx).max(),
#         alpha=1.0, label='Proposal Distribution', lw=2)
# ax.plot(xx, scipy.stats.norm(loc=xs, scale=0.3).pdf(xx)/scipy.stats.norm(loc=x0, scale=0.3).pdf(xx).max(),
#         alpha=0.2, color='k', lw=2)
# ax.plot(x0, scipy.stats.norm(loc=0, scale=1).pdf(x0)/scipy.stats.norm(loc=0, scale=1).pdf(xx).max(), 'o',
#         label='Initial', markersize=10)
# ax.plot(xs, scipy.stats.norm(loc=0, scale=1).pdf(xs)/scipy.stats.norm(loc=0, scale=1).pdf(xx).max(), 'o',
#         label='Proposed', markersize=10)
# ax.axvline(xs, ls='--', color='k', alpha=0.3)
# ax.axvline(x0, ls='--', color='C1', alpha=0.3)
#
# # posterior ratio
# pr = np.exp(simple_gaussian_lnpost(xs) - simple_gaussian_lnpost(x0))
#
# # proposal ratio
# qxy = scipy.stats.norm(loc=xs, scale=0.3).pdf(x0)
# qyx = scipy.stats.norm(loc=x0, scale=0.3).pdf(xs)
# H = (pr * qxy/qyx)[0]
#
# ax.set_title('$\pi(x_*)/\pi(x_0)$ = {:2.2}\n $q(x_0|x_*)/q(x_*|x_0)$ = {:2.2}\n $H$ = {:2.2}'.format(pr[0], (qxy/qyx)[0], H),
#              fontsize=14)
#
#
# ax.set_ylim(ymin=0)
# plt.legend(loc='best', frameon=False)
# ax.set_xlabel(r'$x$', fontsize=14);
#
# plt.show()
#
#
# xx = np.linspace(-4., 4, 1000)
# x0 = np.array([-0.3])
# xs = np.array([1.2])
# fig = plt.figure(figsize=(8,5))
# ax = plt.subplot(111)
#
# ax.plot(xx, scipy.stats.norm(loc=0, scale=1).pdf(xx)/scipy.stats.norm(loc=0, scale=1).pdf(xx).max(),
#         label='Posterior Distribution', lw=2)
# ax.plot(xx, scipy.stats.norm(loc=-1.0, scale=1.0).pdf(xx)/scipy.stats.norm(loc=-1.0, scale=1.0).pdf(xx).max(),
#         alpha=1.0, label='Proposal Distribution', lw=2)
# ax.plot(x0, scipy.stats.norm(loc=0, scale=1).pdf(x0)/scipy.stats.norm(loc=0, scale=1).pdf(xx).max(), 'o',
#         label='Initial', markersize=10)
# ax.plot(xs, scipy.stats.norm(loc=0, scale=1).pdf(xs)/scipy.stats.norm(loc=0, scale=1).pdf(xx).max(), 'o',
#         label='Proposed', markersize=10)
# ax.axvline(xs, ls='--', color='k', alpha=0.3)
# ax.axvline(x0, ls='--', color='C1', alpha=0.3)
#
# # posterior ratio
# pr = np.exp(simple_gaussian_lnpost(xs) - simple_gaussian_lnpost(x0))
#
# # proposal ratio
# qxy = scipy.stats.norm(loc=-1.0, scale=1.0).pdf(x0)
# qyx = scipy.stats.norm(loc=-1.0, scale=1.0).pdf(xs)
# H = (pr * qxy/qyx)[0]
#
# ax.set_title('$\pi(x_*)/\pi(x_0)$ = {:2.2}\n $q(x_0|x_*)/q(x_*|x_0)$ = {:3.3}\n $H$ = {:2.2}'.format(pr[0], (qxy/qyx)[0], H),
#              fontsize=14)
#
#
# ax.set_ylim(ymin=0)
# plt.legend(loc='best', frameon=False)
# ax.set_xlabel(r'$x$', fontsize=14);
#
# plt.show()
#
# xx = np.linspace(-4., 4, 1000)
# x0 = np.array([1.2])
# xs = np.array([-1.2])
# fig = plt.figure(figsize=(8,5))
# ax = plt.subplot(111)
#
# ax.plot(xx, scipy.stats.norm(loc=0, scale=1).pdf(xx)/scipy.stats.norm(loc=0, scale=1).pdf(xx).max(),
#         label='Posterior Distribution', lw=2)
# ax.plot(xx, scipy.stats.norm(loc=-1.0, scale=1.0).pdf(xx)/scipy.stats.norm(loc=-1.0, scale=1.0).pdf(xx).max(),
#         alpha=1.0, label='Proposal Distribution', lw=2)
# ax.plot(x0, scipy.stats.norm(loc=0, scale=1).pdf(x0)/scipy.stats.norm(loc=0, scale=1).pdf(xx).max(), 'o',
#         label='Initial', markersize=10)
# ax.plot(xs, scipy.stats.norm(loc=0, scale=1).pdf(xs)/scipy.stats.norm(loc=0, scale=1).pdf(xx).max(), 'o',
#         label='Proposed', markersize=10)
# ax.axvline(xs, ls='--', color='k', alpha=0.3)
# ax.axvline(x0, ls='--', color='C1', alpha=0.3)
#
# # posterior ratio
# pr = np.exp(simple_gaussian_lnpost(xs) - simple_gaussian_lnpost(x0))
#
# # proposal ratio
# qxy = scipy.stats.norm(loc=-1.0, scale=1.0).pdf(x0)
# qyx = scipy.stats.norm(loc=-1.0, scale=1.0).pdf(xs)
# H = (pr * qxy/qyx)[0]
#
# ax.set_title('$\pi(x_*)/\pi(x_0)$ = {:2.2}\n $q(x_0|x_*)/q(x_*|x_0)$ = {:3.3}\n $H$ = {:2.2}'.format(pr[0], (qxy/qyx)[0], H),
#              fontsize=14)
#
#
# ax.set_ylim(ymin=0)
# plt.legend(loc='best', frameon=False)
# ax.set_xlabel(r'$x$', fontsize=14);
#
# plt.show()


def run_mcmc_plots(sigma):
    x0 = np.random.randn(1)
    chain, ar, lnprob = mh_sampler(x0, simple_gaussian_lnpost, gaussian_proposal,
                                   prop_fn_kwargs={'sigma': sigma})

    plt.figure(figsize=(15, 8))

    burn = int(0.1 * chain.shape[0])

    plt.subplot(221)
    # plt.hist(chain[burn:, 0], 50, normed=True);
    plt.hist(chain[burn:, 0], 50, density=True);
    plt.xlabel(r'$x$', fontsize=15)
    xx = np.linspace(-3.5, 3.5, 1000)
    plt.plot(xx, scipy.stats.norm(loc=0, scale=1).pdf(xx), lw=2)

    plt.subplot(222)
    plt.plot(chain[burn:, 0])
    plt.ylabel(r'$x$', fontsize=15)
    plt.axhline(0.0, lw=2, color='C1')

    plt.subplot(223)
    plt.plot(lnprob[burn:])
    plt.ylabel('log-posterior', fontsize=15)

    plt.subplot(224)
    plt.plot(ar[burn:])
    plt.ylabel('Acceptance Rate', fontsize=15)

    plt.suptitle(r'$\sigma = {}$'.format(sigma), fontsize=15, y=1.02)
    plt.tight_layout()
    plt.show()

run_mcmc_plots(0.01)