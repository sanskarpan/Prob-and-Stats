"""
Parameter Estimation
====================

Maximum Likelihood Estimation (MLE), Maximum A Posteriori (MAP),
and Bayesian inference.

Applications in ML/DL:
- Parameter learning in probabilistic models
- Bayesian neural networks
- Uncertainty quantification
- Prior specification
"""

import math
from typing import List, Tuple, Callable, Optional

try:
    from .statistics import mean, variance
except ImportError:
    # Direct import (non-package context): sys.path must include the module directory
    from statistics import mean, variance  # type: ignore[no-redef]


# ── Private helper: regularized incomplete beta (for exact credible intervals) ─

def _betainc(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function I_x(a, b) via modified Lentz's CF (NR 6.4)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # Use symmetry relation to keep x in the converging half of the CF
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(1.0 - x, b, a)
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - log_beta) / a
    # Modified Lentz (NR betacf): c starts at 1.0, d starts at 1/b_1
    TINY = 1e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < TINY:
        d = TINY
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        # Even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < TINY:
            d = TINY
        c = 1.0 + aa / c
        if abs(c) < TINY:
            c = TINY
        d = 1.0 / d
        h *= d * c
        # Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < TINY:
            d = TINY
        c = 1.0 + aa / c
        if abs(c) < TINY:
            c = TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-9:
            break
    return front * h


def _beta_ppf(p: float, a: float, b: float) -> float:
    """Quantile of Beta(a, b) at probability p via binary search on _betainc."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _betainc(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def maximum_likelihood(
    data: List[float],
    distribution: str = 'normal'
) -> dict:
    """
    Maximum Likelihood Estimation for common distributions.

    MLE finds parameters that maximize P(data | parameters).

    Args:
        data: Observed data
        distribution: Distribution type ('normal', 'bernoulli', 'poisson', 'exponential')

    Returns:
        Dictionary of estimated parameters

    Example:
        >>> data = [1.0, 2.0, 3.0, 4.0, 5.0]
        >>> mle = maximum_likelihood(data, 'normal')
        >>> mle['mu']
        3.0
    """
    if distribution == 'normal':
        # MLE for normal: μ = sample mean, σ² = sample variance
        mu = mean(data)
        sigma_squared = variance(data, ddof=0)  # MLE uses population variance
        return {'mu': mu, 'sigma': math.sqrt(sigma_squared)}

    elif distribution == 'bernoulli':
        # MLE for Bernoulli: p = (# successes) / n
        # Assume data is 0s and 1s
        p = sum(data) / len(data)
        return {'p': p}

    elif distribution == 'poisson':
        # MLE for Poisson: λ = sample mean
        lam = mean(data)
        return {'lambda': lam}

    elif distribution == 'exponential':
        # MLE for Exponential: λ = 1 / sample mean
        lam = 1.0 / mean(data)
        return {'lambda': lam}

    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def maximum_a_posteriori(
    data: List[float],
    prior_params: dict,
    distribution: str = 'normal'
) -> dict:
    """
    Maximum A Posteriori estimation.

    MAP finds parameters that maximize P(parameters | data) ∝ P(data | parameters) * P(parameters)

    Args:
        data: Observed data
        prior_params: Prior parameters
        distribution: Distribution type

    Returns:
        Dictionary of MAP estimates

    Example:
        >>> data = [1.0, 2.0, 3.0, 4.0, 5.0]
        >>> prior = {'mu_prior': 2.0, 'sigma_prior': 1.0, 'precision': 1.0}
        >>> map_est = maximum_a_posteriori(data, prior, 'normal')
    """
    if distribution == 'normal':
        # Assuming prior on μ is Normal(μ₀, σ₀²)
        # Posterior mean: (τ₀μ₀ + nτx̄) / (τ₀ + nτ)
        # where τ = 1/σ² is precision

        mu_prior = prior_params.get('mu_prior', 0.0)
        tau_prior = prior_params.get('precision', 1.0)  # Prior precision

        n = len(data)
        x_bar = mean(data)
        sigma = prior_params.get('sigma', 1.0)  # Assumed known
        tau = 1.0 / (sigma ** 2)

        # MAP estimate for μ
        mu_map = (tau_prior * mu_prior + n * tau * x_bar) / (tau_prior + n * tau)

        return {'mu': mu_map, 'sigma': sigma}

    elif distribution == 'bernoulli':
        # Beta prior on p: Beta(α, β)
        # Posterior: Beta(α + # successes, β + # failures)
        # MAP: (α + # successes - 1) / (α + β + n - 2)

        alpha = prior_params.get('alpha', 1.0)
        beta = prior_params.get('beta', 1.0)

        successes = sum(data)
        n = len(data)

        # MAP estimate
        if alpha + beta + n - 2 > 0:
            p_map = (alpha + successes - 1) / (alpha + beta + n - 2)
        else:
            p_map = successes / n  # Fall back to MLE

        return {'p': max(0.0, min(1.0, p_map))}

    else:
        raise ValueError(f"MAP not implemented for {distribution}")


def bayesian_update(
    prior_params: dict,
    data: List[float],
    distribution: str = 'normal'
) -> dict:
    """
    Bayesian update - compute posterior parameters.

    prior × likelihood → posterior

    Args:
        prior_params: Prior distribution parameters
        data: Observed data
        distribution: Distribution type

    Returns:
        Posterior distribution parameters

    Example:
        >>> prior = {'mu': 0.0, 'sigma': 1.0, 'precision': 1.0}
        >>> data = [1.0, 2.0, 3.0]
        >>> posterior = bayesian_update(prior, data, 'normal')
    """
    if distribution == 'normal':
        # Conjugate prior: Normal prior on μ with known σ
        # Prior: μ ~ N(μ₀, σ₀²)
        # Likelihood: X ~ N(μ, σ²) with known σ
        # Posterior: μ ~ N(μₙ, σₙ²)

        mu_0 = prior_params.get('mu', 0.0)
        tau_0 = prior_params.get('precision', 1.0)  # 1/σ₀²
        sigma = prior_params.get('sigma', 1.0)  # Known σ

        n = len(data)
        x_bar = mean(data)
        tau = 1.0 / (sigma ** 2)

        # Posterior parameters
        tau_n = tau_0 + n * tau
        mu_n = (tau_0 * mu_0 + n * tau * x_bar) / tau_n
        sigma_n = 1.0 / math.sqrt(tau_n)

        return {'mu': mu_n, 'sigma': sigma_n, 'precision': tau_n}

    elif distribution == 'bernoulli':
        # Conjugate prior: Beta prior on p
        # Prior: p ~ Beta(α₀, β₀)
        # Likelihood: X ~ Bernoulli(p)
        # Posterior: p ~ Beta(αₙ, βₙ)

        alpha_0 = prior_params.get('alpha', 1.0)
        beta_0 = prior_params.get('beta', 1.0)

        successes = sum(data)
        failures = len(data) - successes

        # Posterior parameters
        alpha_n = alpha_0 + successes
        beta_n = beta_0 + failures

        # Posterior mean
        posterior_mean = alpha_n / (alpha_n + beta_n)

        return {
            'alpha': alpha_n,
            'beta': beta_n,
            'mean': posterior_mean,
            'variance': (alpha_n * beta_n) / ((alpha_n + beta_n)**2 * (alpha_n + beta_n + 1))
        }

    elif distribution == 'poisson':
        # Conjugate prior: Gamma prior on λ
        # Prior: λ ~ Gamma(α₀, β₀)
        # Likelihood: X ~ Poisson(λ)
        # Posterior: λ ~ Gamma(αₙ, βₙ)

        alpha_0 = prior_params.get('alpha', 1.0)
        beta_0 = prior_params.get('beta', 1.0)

        n = len(data)
        sum_x = sum(data)

        # Posterior parameters
        alpha_n = alpha_0 + sum_x
        beta_n = beta_0 + n

        # Posterior mean
        posterior_mean = alpha_n / beta_n

        return {
            'alpha': alpha_n,
            'beta': beta_n,
            'mean': posterior_mean,
            'variance': alpha_n / (beta_n ** 2)
        }

    else:
        raise ValueError(f"Bayesian update not implemented for {distribution}")


def posterior_predictive(
    posterior_params: dict,
    distribution: str = 'normal',
    n_samples: int = 1000
) -> List[float]:
    """
    Generate samples from posterior predictive distribution.

    Integrates over parameter uncertainty to make predictions.

    Args:
        posterior_params: Posterior distribution parameters
        distribution: Distribution type
        n_samples: Number of samples to generate

    Returns:
        List of samples from posterior predictive

    Example:
        >>> posterior = {'mu': 3.0, 'sigma': 1.0}
        >>> samples = posterior_predictive(posterior, 'normal', n_samples=100)
    """
    import random

    if distribution == 'normal':
        # For conjugate Normal-Normal with known variance:
        # Posterior predictive is Normal(μₙ, σ² + σₙ²)
        mu = posterior_params['mu']
        sigma = posterior_params['sigma']

        # In full Bayesian setting with unknown variance, this would be Student-t
        # Here using simplified version
        samples = [random.gauss(mu, sigma) for _ in range(n_samples)]
        return samples

    elif distribution == 'bernoulli':
        # Beta-Bernoulli: posterior predictive is Bernoulli with marginal probability
        alpha = posterior_params['alpha']
        beta = posterior_params['beta']

        # Posterior mean of p
        p = alpha / (alpha + beta)

        samples = [1 if random.random() < p else 0 for _ in range(n_samples)]
        return samples

    else:
        raise ValueError(f"Posterior predictive not implemented for {distribution}")


def credible_interval(
    posterior_params: dict,
    distribution: str = 'normal',
    credibility: float = 0.95
) -> Tuple[float, float]:
    """
    Compute Bayesian credible interval (analogous to confidence interval).

    Args:
        posterior_params: Posterior distribution parameters
        distribution: Distribution type
        credibility: Credibility level (e.g., 0.95 for 95%)

    Returns:
        Tuple of (lower bound, upper bound)

    Example:
        >>> posterior = {'mu': 3.0, 'sigma': 1.0}
        >>> ci = credible_interval(posterior, 'normal', 0.95)
    """
    if distribution == 'normal':
        mu = posterior_params['mu']
        sigma = posterior_params['sigma']

        # For Normal distribution, credible interval is
        # [μ - z*σ, μ + z*σ]
        z_values = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_values.get(credibility, 1.96)

        lower = mu - z * sigma
        upper = mu + z * sigma

        return (lower, upper)

    elif distribution == 'bernoulli':
        # Exact Beta quantiles via regularized incomplete beta function
        alpha = posterior_params['alpha']
        beta = posterior_params['beta']
        tail = (1.0 - credibility) / 2.0
        lower = _beta_ppf(tail, alpha, beta)
        upper = _beta_ppf(1.0 - tail, alpha, beta)
        return (lower, upper)

    else:
        raise ValueError(f"Credible interval not implemented for {distribution}")


def empirical_bayes(
    data_groups: List[List[float]],
    distribution: str = 'normal'
) -> dict:
    """
    Empirical Bayes estimation - estimate prior from data.

    Args:
        data_groups: List of data groups
        distribution: Distribution type

    Returns:
        Estimated prior parameters

    Example:
        >>> groups = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        >>> prior = empirical_bayes(groups, 'normal')
    """
    if distribution == 'normal':
        # Estimate prior mean and variance from group means
        group_means = [mean(group) for group in data_groups]

        prior_mu = mean(group_means)
        prior_sigma = math.sqrt(variance(group_means, ddof=1))

        return {'mu': prior_mu, 'sigma': prior_sigma, 'precision': 1.0 / (prior_sigma**2)}

    else:
        raise ValueError(f"Empirical Bayes not implemented for {distribution}")


def likelihood_ratio(
    data: List[float],
    params1: dict,
    params2: dict,
    distribution: str = 'normal'
) -> float:
    """
    Compute likelihood ratio for model comparison.

    LR = P(data | model1) / P(data | model2)

    Args:
        data: Observed data
        params1: Parameters for model 1
        params2: Parameters for model 2
        distribution: Distribution type

    Returns:
        Likelihood ratio

    Example:
        >>> data = [1.0, 2.0, 3.0]
        >>> p1 = {'mu': 2.0, 'sigma': 1.0}
        >>> p2 = {'mu': 5.0, 'sigma': 1.0}
        >>> lr = likelihood_ratio(data, p1, p2, 'normal')
    """
    if distribution == 'normal':
        mu1, sigma1 = params1['mu'], params1['sigma']
        mu2, sigma2 = params2['mu'], params2['sigma']

        # Log-likelihood to avoid numerical issues
        log_lik1 = sum(_normal_log_likelihood(x, mu1, sigma1) for x in data)
        log_lik2 = sum(_normal_log_likelihood(x, mu2, sigma2) for x in data)

        return math.exp(log_lik1 - log_lik2)

    else:
        raise ValueError(f"Likelihood ratio not implemented for {distribution}")


def _normal_log_likelihood(x: float, mu: float, sigma: float) -> float:
    """Helper: compute log-likelihood for single normal observation."""
    return -0.5 * math.log(2 * math.pi * sigma**2) - 0.5 * ((x - mu) / sigma)**2


def bootstrap_parameter_estimate(
    data: List[float],
    estimator_func: Callable,
    n_bootstrap: int = 1000
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Bootstrap estimation with confidence interval.

    Args:
        data: Original data
        estimator_func: Function to estimate parameter (e.g., mean, median)
        n_bootstrap: Number of bootstrap samples

    Returns:
        Tuple of (estimate, std_error, confidence_interval)

    Example:
        >>> from .statistics import median
        >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> est, se, ci = bootstrap_parameter_estimate(data, median)
    """
    import random

    n = len(data)
    bootstrap_estimates = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = [data[random.randint(0, n-1)] for _ in range(n)]
        bootstrap_estimates.append(estimator_func(sample))

    # Point estimate (original data)
    estimate = estimator_func(data)

    # Standard error (std of bootstrap estimates)
    bootstrap_mean = mean(bootstrap_estimates)
    bootstrap_var = sum((x - bootstrap_mean)**2 for x in bootstrap_estimates) / (n_bootstrap - 1)
    std_error = math.sqrt(bootstrap_var)

    # 95% confidence interval (percentile method)
    sorted_estimates = sorted(bootstrap_estimates)
    lower_idx = int(0.025 * n_bootstrap)
    upper_idx = int(0.975 * n_bootstrap)
    ci = (sorted_estimates[lower_idx], sorted_estimates[upper_idx])

    return estimate, std_error, ci


def conjugate_prior_parameters(distribution: str) -> dict:
    """
    Get default conjugate prior parameters for common distributions.

    Args:
        distribution: Distribution type

    Returns:
        Dictionary of default prior parameters

    Example:
        >>> conjugate_prior_parameters('bernoulli')
        {'alpha': 1.0, 'beta': 1.0}
    """
    priors = {
        'bernoulli': {'alpha': 1.0, 'beta': 1.0},  # Beta(1,1) = Uniform
        'poisson': {'alpha': 1.0, 'beta': 1.0},     # Gamma(1,1)
        'normal': {'mu': 0.0, 'sigma': 1.0, 'precision': 1.0},  # N(0,1)
        'exponential': {'alpha': 1.0, 'beta': 1.0}  # Gamma(1,1)
    }

    if distribution not in priors:
        raise ValueError(f"No default prior for {distribution}")

    return priors[distribution]
