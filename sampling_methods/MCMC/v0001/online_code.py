import numpy as np
from scipy.special import gamma, factorial
from matplotlib import pyplot as plt
breakpoint()

# Prior alpha = 7, beta = 1
# Start with a value of lambda given by 8.0 and compute the prior probability density of observing this value

def prior_prob_density(lam, alpha, beta):
    return (beta ** (alpha) * lam ** (alpha - 1) * np.exp(-beta * lam) / gamma(alpha))


def likelihood_density(data, lam):
    return (lam ** (data) * np.exp(-lam) / factorial(data))


# Starting value of lambda
lambda_current = 8.0
# Prior parameters alpha and beta
alpha = 7.0
beta = 1.0
# Observed data of 9 outages
data_val = 9

lambda_array = np.zeros(1000)

for i in range(1000):

    # Current value
    prior = prior_prob_density(lam=lambda_current, alpha=alpha, beta=beta)
    likelihood = likelihood_density(data=data_val, lam=lambda_current)
    posterior_current = likelihood * prior

    # Proposed value
    lambda_proposed = np.random.normal(lambda_current, scale=0.5)  # scale is our tuning parameter
    prior = prior_prob_density(lam=lambda_proposed, alpha=alpha, beta=beta)
    likelihood = likelihood_density(data=data_val, lam=lambda_proposed)
    posterior_proposed = likelihood * prior

    # Compute the probability of move
    ratio = posterior_proposed / posterior_current
    p_move = min(ratio, 1)
    random_draw = np.random.uniform(0, 1)
    if (random_draw < p_move):
        lambda_current = lambda_proposed

    # Store the current value
    lambda_array[i] = lambda_current

plt.hist(lambda_array)
plt.show()








