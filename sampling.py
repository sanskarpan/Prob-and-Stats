"""
Sampling Methods
================

Monte Carlo, MCMC, Gibbs sampling, Metropolis-Hastings, and more.

Applications in ML/DL:
- Bayesian inference
- Variational inference
- Generative models
- Uncertainty quantification
- Approximate inference
"""

import random
import math
from typing import List, Callable, Tuple, Optional


def monte_carlo_sample(
    distribution_func: Callable[[], float],
    n_samples: int = 1000
) -> List[float]:
    """
    Basic Monte Carlo sampling.

    Args:
        distribution_func: Function that generates samples from distribution
        n_samples: Number of samples to generate

    Returns:
        List of samples

    Example:
        >>> import random
        >>> samples = monte_carlo_sample(lambda: random.gauss(0, 1), 1000)
    """
    return [distribution_func() for _ in range(n_samples)]


def monte_carlo_integration(
    func: Callable[[float], float],
    a: float,
    b: float,
    n_samples: int = 10000
) -> float:
    """
    Monte Carlo integration to estimate ∫ f(x) dx over [a, b].

    Args:
        func: Function to integrate
        a: Lower bound
        b: Upper bound
        n_samples: Number of samples

    Returns:
        Estimated integral

    Example:
        >>> # Estimate ∫₀¹ x² dx = 1/3
        >>> result = monte_carlo_integration(lambda x: x**2, 0, 1, 10000)
    """
    samples = [random.uniform(a, b) for _ in range(n_samples)]
    func_values = [func(x) for x in samples]
    return (b - a) * sum(func_values) / n_samples


def inverse_transform_sampling(
    inverse_cdf: Callable[[float], float],
    n_samples: int = 1000
) -> List[float]:
    """
    Inverse transform sampling using inverse CDF.

    If U ~ Uniform(0,1), then F⁻¹(U) ~ F

    Args:
        inverse_cdf: Inverse cumulative distribution function
        n_samples: Number of samples

    Returns:
        List of samples

    Example:
        >>> # Exponential distribution with λ=1
        >>> inv_cdf = lambda u: -math.log(1 - u)
        >>> samples = inverse_transform_sampling(inv_cdf, 1000)
    """
    uniform_samples = [random.random() for _ in range(n_samples)]
    return [inverse_cdf(u) for u in uniform_samples]


def rejection_sampling(
    target_pdf: Callable[[float], float],
    proposal_sampler: Callable[[], float],
    proposal_pdf: Callable[[float], float],
    M: float,
    n_samples: int = 1000,
    max_iterations: int = 100000
) -> List[float]:
    """
    Rejection sampling (accept-reject method).

    Sample from target distribution using proposal distribution.
    Requires: target_pdf(x) ≤ M * proposal_pdf(x) for all x

    Args:
        target_pdf: Target probability density function
        proposal_sampler: Function to sample from proposal distribution
        proposal_pdf: Proposal probability density function
        M: Constant such that target ≤ M * proposal
        n_samples: Number of samples to generate
        max_iterations: Maximum iterations to prevent infinite loops

    Returns:
        List of accepted samples

    Example:
        >>> # Sample from Beta(2,2) using Uniform(0,1)
        >>> target = lambda x: 6 * x * (1-x)
        >>> proposal = lambda: random.random()
        >>> proposal_pdf = lambda x: 1.0
        >>> samples = rejection_sampling(target, proposal, proposal_pdf, M=1.5, n_samples=100)
    """
    samples = []
    iterations = 0

    while len(samples) < n_samples and iterations < max_iterations:
        # Sample from proposal
        x = proposal_sampler()

        # Acceptance probability
        u = random.random()
        acceptance_prob = target_pdf(x) / (M * proposal_pdf(x))

        if u <= acceptance_prob:
            samples.append(x)

        iterations += 1

    if len(samples) < n_samples:
        print(f"Warning: Only generated {len(samples)} samples after {iterations} iterations")

    return samples


def importance_sampling(
    func: Callable[[float], float],
    target_pdf: Callable[[float], float],
    proposal_sampler: Callable[[], float],
    proposal_pdf: Callable[[float], float],
    n_samples: int = 1000
) -> Tuple[float, float]:
    """
    Importance sampling for expectation estimation.

    Estimate E_p[f(X)] using samples from proposal distribution q.

    E_p[f(X)] ≈ (1/n) Σ f(xᵢ) * p(xᵢ) / q(xᵢ)

    Args:
        func: Function to compute expectation of
        target_pdf: Target distribution p(x)
        proposal_sampler: Sampler for proposal distribution q(x)
        proposal_pdf: Proposal distribution density q(x)
        n_samples: Number of samples

    Returns:
        Tuple of (estimate, effective_sample_size)

    Example:
        >>> # Estimate E[x²] under Normal(0,1) using samples from Normal(0,2)
        >>> f = lambda x: x**2
        >>> target = lambda x: (1/math.sqrt(2*math.pi)) * math.exp(-x**2/2)
        >>> proposal_sample = lambda: random.gauss(0, 2)
        >>> proposal = lambda x: (1/math.sqrt(8*math.pi)) * math.exp(-x**2/8)
        >>> est, ess = importance_sampling(f, target, proposal_sample, proposal, 1000)
    """
    # Generate samples from proposal
    samples = [proposal_sampler() for _ in range(n_samples)]

    # Compute importance weights
    weights = [target_pdf(x) / proposal_pdf(x) for x in samples]

    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Weighted average
    estimate = sum(func(x) * w for x, w in zip(samples, normalized_weights))

    # Effective sample size
    ess = 1.0 / sum(w**2 for w in normalized_weights)

    return estimate, ess


def metropolis_hastings(
    target_log_pdf: Callable[[float], float],
    proposal_sampler: Callable[[float], float],
    initial_state: float,
    n_samples: int = 10000,
    burn_in: int = 1000,
    thin: int = 1
) -> List[float]:
    """
    Metropolis-Hastings MCMC algorithm.

    Sample from target distribution using Markov Chain Monte Carlo.

    Args:
        target_log_pdf: Log of target probability density
        proposal_sampler: Function x_new = f(x_current) for proposals
        initial_state: Starting state for chain
        n_samples: Number of samples to generate (after burn-in and thinning)
        burn_in: Number of initial samples to discard
        thin: Keep every nth sample (reduces autocorrelation)

    Returns:
        List of samples from target distribution

    Example:
        >>> # Sample from Normal(0, 1)
        >>> target = lambda x: -0.5 * x**2
        >>> proposal = lambda x: x + random.gauss(0, 0.5)
        >>> samples = metropolis_hastings(target, proposal, 0.0, 1000)
    """
    samples = []
    current_state = initial_state
    current_log_prob = target_log_pdf(current_state)

    total_iterations = burn_in + n_samples * thin
    accepted = 0

    for i in range(total_iterations):
        # Propose new state
        proposed_state = proposal_sampler(current_state)
        proposed_log_prob = target_log_pdf(proposed_state)

        # Acceptance ratio (log scale for numerical stability)
        log_acceptance_ratio = proposed_log_prob - current_log_prob

        # Accept or reject (guard against log(0) when random() returns 0.0)
        if math.log(random.random() + 1e-300) < log_acceptance_ratio:
            current_state = proposed_state
            current_log_prob = proposed_log_prob
            accepted += 1

        # Store sample (after burn-in, with thinning)
        if i >= burn_in and (i - burn_in) % thin == 0:
            samples.append(current_state)

    acceptance_rate = accepted / total_iterations
    if acceptance_rate < 0.1 or acceptance_rate > 0.9:
        print(f"Warning: Acceptance rate = {acceptance_rate:.2f} (typical range: 0.1-0.9)")

    return samples


def gibbs_sampling(
    conditional_samplers: List[Callable[[List[float]], float]],
    initial_state: List[float],
    n_samples: int = 10000,
    burn_in: int = 1000,
    thin: int = 1
) -> List[List[float]]:
    """
    Gibbs sampling for multivariate distributions.

    Sample each variable conditionally on all others.

    Args:
        conditional_samplers: List of functions to sample each variable
                            conditionally on others
        initial_state: Initial state for all variables
        n_samples: Number of samples to generate
        burn_in: Number of initial samples to discard
        thin: Thinning interval

    Returns:
        List of multivariate samples

    Example:
        >>> # Sample from bivariate Normal
        >>> # x | y ~ N(0.5*y, 0.75)
        >>> # y | x ~ N(0.5*x, 0.75)
        >>> samplers = [
        ...     lambda state: random.gauss(0.5 * state[1], math.sqrt(0.75)),
        ...     lambda state: random.gauss(0.5 * state[0], math.sqrt(0.75))
        ... ]
        >>> samples = gibbs_sampling(samplers, [0.0, 0.0], n_samples=1000)
    """
    n_vars = len(conditional_samplers)
    samples = []
    current_state = list(initial_state)

    total_iterations = burn_in + n_samples * thin

    for i in range(total_iterations):
        # Update each variable in turn
        for j in range(n_vars):
            current_state[j] = conditional_samplers[j](current_state)

        # Store sample (after burn-in, with thinning)
        if i >= burn_in and (i - burn_in) % thin == 0:
            samples.append(list(current_state))

    return samples


def slice_sampling(
    log_pdf: Callable[[float], float],
    initial_state: float,
    width: float = 1.0,
    n_samples: int = 1000,
    burn_in: int = 100
) -> List[float]:
    """
    Slice sampling - auxiliary variable MCMC method.

    Args:
        log_pdf: Log probability density function
        initial_state: Starting point
        width: Initial width for finding slice
        n_samples: Number of samples to generate
        burn_in: Burn-in period

    Returns:
        List of samples

    Example:
        >>> # Sample from Normal(0, 1)
        >>> log_pdf = lambda x: -0.5 * x**2
        >>> samples = slice_sampling(log_pdf, 0.0, width=2.0, n_samples=1000)
    """
    samples = []
    x = initial_state

    for i in range(burn_in + n_samples):
        # Sample vertical level uniformly
        log_y = log_pdf(x) + math.log(random.random())

        # Find slice (interval where log_pdf(x') > log_y)
        # Using stepping out procedure
        left = x - width * random.random()
        right = left + width

        # Step out (capped to prevent infinite loop on flat/improper targets)
        max_steps = 1000
        steps = 0
        while log_pdf(left) > log_y and steps < max_steps:
            left -= width
            steps += 1
        steps = 0
        while log_pdf(right) > log_y and steps < max_steps:
            right += width
            steps += 1

        # Sample from slice using shrinkage
        while True:
            x_new = random.uniform(left, right)
            if log_pdf(x_new) > log_y:
                x = x_new
                break
            else:
                # Shrink interval
                if x_new < x:
                    left = x_new
                else:
                    right = x_new

        if i >= burn_in:
            samples.append(x)

    return samples


def hamiltonian_monte_carlo(
    log_pdf: Callable[[float], float],
    grad_log_pdf: Callable[[float], float],
    initial_state: float,
    epsilon: float = 0.1,
    L: int = 10,
    n_samples: int = 1000,
    burn_in: int = 100
) -> List[float]:
    """
    Hamiltonian Monte Carlo (HMC) - uses gradient information.

    More efficient than Metropolis-Hastings for complex distributions.

    Args:
        log_pdf: Log probability density
        grad_log_pdf: Gradient of log probability
        initial_state: Starting point
        epsilon: Step size for leapfrog integrator
        L: Number of leapfrog steps
        n_samples: Number of samples
        burn_in: Burn-in period

    Returns:
        List of samples

    Example:
        >>> # Sample from Normal(0, 1)
        >>> log_pdf = lambda x: -0.5 * x**2
        >>> grad = lambda x: -x
        >>> samples = hamiltonian_monte_carlo(log_pdf, grad, 0.0, n_samples=1000)
    """
    samples = []
    q = initial_state

    accepted = 0

    for i in range(burn_in + n_samples):
        # Sample momentum
        p = random.gauss(0, 1)

        # Store current state
        q_current = q
        p_current = p

        # Leapfrog integration
        p = p + 0.5 * epsilon * grad_log_pdf(q)

        for _ in range(L):
            q = q + epsilon * p
            if _ != L - 1:  # Don't do full step at end
                p = p + epsilon * grad_log_pdf(q)

        p = p + 0.5 * epsilon * grad_log_pdf(q)

        # Negate momentum for reversibility
        p = -p

        # Compute acceptance probability
        current_H = -log_pdf(q_current) + 0.5 * p_current**2
        proposed_H = -log_pdf(q) + 0.5 * p**2

        if random.random() < math.exp(current_H - proposed_H):
            # Accept
            accepted += 1
        else:
            # Reject - keep current state
            q = q_current

        if i >= burn_in:
            samples.append(q)

    acceptance_rate = accepted / (burn_in + n_samples)
    if acceptance_rate < 0.5 or acceptance_rate > 0.999:
        print(f"Warning: HMC acceptance rate = {acceptance_rate:.2f} (typical: 0.5-0.999)")

    return samples


def langevin_dynamics(
    grad_log_pdf: Callable[[float], float],
    initial_state: float,
    epsilon: float = 0.01,
    n_samples: int = 1000,
    burn_in: int = 100
) -> List[float]:
    """
    Langevin dynamics (gradient-based MCMC).

    Uses gradient to guide proposals: x_new = x + ε∇log p(x) + √(2ε) * noise

    Args:
        grad_log_pdf: Gradient of log probability
        initial_state: Starting point
        epsilon: Step size
        n_samples: Number of samples
        burn_in: Burn-in period

    Returns:
        List of samples

    Example:
        >>> # Sample from Normal(0, 1)
        >>> grad = lambda x: -x
        >>> samples = langevin_dynamics(grad, 0.0, epsilon=0.1, n_samples=1000)
    """
    samples = []
    x = initial_state

    for i in range(burn_in + n_samples):
        # Langevin update
        noise = random.gauss(0, 1)
        x = x + epsilon * grad_log_pdf(x) + math.sqrt(2 * epsilon) * noise

        if i >= burn_in:
            samples.append(x)

    return samples


def parallel_tempering(
    log_pdf: Callable[[float], float],
    initial_states: List[float],
    temperatures: List[float],
    proposal_sampler: Callable[[float], float],
    n_samples: int = 1000,
    burn_in: int = 100,
    swap_interval: int = 10
) -> List[float]:
    """
    Parallel tempering (replica exchange MCMC).

    Runs multiple chains at different temperatures and swaps states.
    Helps escape local modes.

    Args:
        log_pdf: Log probability density
        initial_states: Initial state for each temperature
        temperatures: Temperature ladder (1.0 for target distribution)
        proposal_sampler: Proposal function
        n_samples: Number of samples from T=1 chain
        burn_in: Burn-in period
        swap_interval: How often to attempt swaps

    Returns:
        Samples from target distribution (T=1)

    Example:
        >>> log_pdf = lambda x: -0.5 * x**2
        >>> proposal = lambda x: x + random.gauss(0, 0.5)
        >>> temps = [1.0, 2.0, 4.0]
        >>> states = [0.0, 0.0, 0.0]
        >>> samples = parallel_tempering(log_pdf, states, temps, proposal, n_samples=1000)
    """
    n_chains = len(temperatures)
    states = list(initial_states)
    samples = []

    for i in range(burn_in + n_samples):
        # Update each chain
        for j in range(n_chains):
            proposed = proposal_sampler(states[j])
            current_log_prob = log_pdf(states[j]) / temperatures[j]
            proposed_log_prob = log_pdf(proposed) / temperatures[j]

            if math.log(random.random() + 1e-300) < (proposed_log_prob - current_log_prob):
                states[j] = proposed

        # Attempt swaps
        if i % swap_interval == 0:
            # Randomly select adjacent chains to swap
            for j in range(n_chains - 1):
                if random.random() < 0.5:
                    # Compute swap acceptance
                    log_prob_j = log_pdf(states[j])
                    log_prob_jp1 = log_pdf(states[j+1])

                    log_ratio = (
                        (log_prob_j / temperatures[j+1] + log_prob_jp1 / temperatures[j]) -
                        (log_prob_j / temperatures[j] + log_prob_jp1 / temperatures[j+1])
                    )

                    if math.log(random.random() + 1e-300) < log_ratio:
                        # Swap
                        states[j], states[j+1] = states[j+1], states[j]

        # Collect samples from T=1 chain (assumed to be first)
        if i >= burn_in:
            samples.append(states[0])

    return samples


def adaptive_metropolis(
    target_log_pdf: Callable[[List[float]], float],
    initial_state: List[float],
    n_samples: int = 10000,
    burn_in: int = 1000,
    adaptation_interval: int = 100
) -> List[List[float]]:
    """
    Adaptive Metropolis - automatically tunes proposal covariance.

    Args:
        target_log_pdf: Log PDF of target distribution
        initial_state: Initial state vector
        n_samples: Number of samples to generate
        burn_in: Burn-in period
        adaptation_interval: Update covariance every N iterations

    Returns:
        List of samples

    Example:
        >>> log_pdf = lambda x: -0.5 * sum(xi**2 for xi in x)
        >>> samples = adaptive_metropolis(log_pdf, [0.0, 0.0], n_samples=1000)
    """
    d = len(initial_state)
    samples = []
    current_state = list(initial_state)
    current_log_prob = target_log_pdf(current_state)

    # Per-dimension proposal scales (diagonal covariance): (2.38²/d) * σ_j²
    # Initialise to identity scaling
    proposal_scales = [2.38**2 / d] * d

    history = []

    for i in range(burn_in + n_samples):
        # Propose new state using per-dimension scales
        noise = [random.gauss(0, 1) for _ in range(d)]
        proposed_state = [current_state[j] + math.sqrt(proposal_scales[j]) * noise[j]
                          for j in range(d)]

        proposed_log_prob = target_log_pdf(proposed_state)

        # Accept/reject (guard against log(0))
        if math.log(random.random() + 1e-300) < (proposed_log_prob - current_log_prob):
            current_state = proposed_state
            current_log_prob = proposed_log_prob

        history.append(list(current_state))

        # Adapt per-dimension proposal scales
        if i > 0 and i % adaptation_interval == 0 and len(history) > d + 1:
            n_hist = len(history)
            dim_means = [sum(s[j] for s in history) / n_hist for j in range(d)]
            # Per-dimension sample variance
            dim_vars = [
                sum((s[j] - dim_means[j]) ** 2 for s in history) / (n_hist - 1)
                for j in range(d)
            ]
            # Haario et al. (2001) optimal scaling: (2.38²/d) * σ_j²
            proposal_scales = [2.38**2 * max(v, 1e-8) / d for v in dim_vars]

        if i >= burn_in:
            samples.append(list(current_state))

    return samples
