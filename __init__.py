"""
Probability and Statistics Module
===================================

A comprehensive implementation of probability theory and statistics from scratch.
Covers distributions, sampling, inference, information theory, and more.

Modules:
--------
- probability: Basic probability theory, Bayes theorem, conditional probability
- distributions: Common probability distributions (Bernoulli, Normal, etc.)
- statistics: Descriptive statistics, hypothesis testing, inference
- information_theory: Entropy, KL divergence, mutual information
- estimation: MLE, MAP, Bayesian inference
- sampling: Monte Carlo, MCMC, Gibbs sampling, Metropolis-Hastings
"""

__version__ = "1.0.0"

# Relative imports only work when this file is loaded as part of a proper
# Python package (i.e. __package__ is set).  Pytest 9+ imports __init__.py
# as a standalone module during package-setup introspection; in that context
# __package__ is '' / None, and relative imports raise ImportError.
if __package__:
    from .probability import (
        conditional_probability,
        bayes_theorem,
        bayes_theorem_with_partitions,
        law_of_total_probability,
        are_independent,
        joint_probability_independent,
        complement,
        union_probability,
        odds_to_probability,
        probability_to_odds,
        bayes_factor,
        posterior_odds,
    )

    from .distributions import (
        Bernoulli,
        Binomial,
        Poisson,
        Categorical,
        Uniform,
        Normal,
        Exponential,
        Beta,
        Gamma,
    )

    from .statistics import (
        mean,
        median,
        mode,
        quantile,
        variance,
        std,
        covariance,
        correlation,
        z_score,
        standard_error,
        confidence_interval,
        t_statistic,
        t_test,
        two_sample_t_test,
        chi_square_test,
        z_test,
        p_value_from_z,
        effect_size_cohens_d,
        ab_test,
        bootstrap_confidence_interval,
    )

    from .information_theory import (
        entropy,
        joint_entropy,
        conditional_entropy,
        cross_entropy,
        kl_divergence,
        js_divergence,
        mutual_information,
        information_gain,
        perplexity,
        binary_cross_entropy,
        categorical_cross_entropy,
        gini_impurity,
        relative_entropy,
    )

    from .estimation import (
        maximum_likelihood,
        maximum_a_posteriori,
        bayesian_update,
        posterior_predictive,
        credible_interval,
        empirical_bayes,
        likelihood_ratio,
        bootstrap_parameter_estimate,
        conjugate_prior_parameters,
    )

    from .sampling import (
        monte_carlo_sample,
        monte_carlo_integration,
        inverse_transform_sampling,
        rejection_sampling,
        importance_sampling,
        metropolis_hastings,
        gibbs_sampling,
        slice_sampling,
        hamiltonian_monte_carlo,
        langevin_dynamics,
        parallel_tempering,
        adaptive_metropolis,
    )

__all__ = [
    # Probability
    'conditional_probability',
    'bayes_theorem',
    'bayes_theorem_with_partitions',
    'law_of_total_probability',
    'are_independent',
    'joint_probability_independent',
    'complement',
    'union_probability',
    'odds_to_probability',
    'probability_to_odds',
    'bayes_factor',
    'posterior_odds',

    # Distributions
    'Bernoulli',
    'Binomial',
    'Poisson',
    'Categorical',
    'Uniform',
    'Normal',
    'Exponential',
    'Beta',
    'Gamma',

    # Statistics
    'mean',
    'median',
    'mode',
    'quantile',
    'variance',
    'std',
    'covariance',
    'correlation',
    'z_score',
    'standard_error',
    'confidence_interval',
    't_statistic',
    't_test',
    'two_sample_t_test',
    'chi_square_test',
    'z_test',
    'p_value_from_z',
    'effect_size_cohens_d',
    'ab_test',
    'bootstrap_confidence_interval',

    # Information Theory
    'entropy',
    'joint_entropy',
    'conditional_entropy',
    'cross_entropy',
    'kl_divergence',
    'js_divergence',
    'mutual_information',
    'information_gain',
    'perplexity',
    'binary_cross_entropy',
    'categorical_cross_entropy',
    'gini_impurity',
    'relative_entropy',

    # Estimation
    'maximum_likelihood',
    'maximum_a_posteriori',
    'bayesian_update',
    'posterior_predictive',
    'credible_interval',
    'empirical_bayes',
    'likelihood_ratio',
    'bootstrap_parameter_estimate',
    'conjugate_prior_parameters',

    # Sampling
    'monte_carlo_sample',
    'monte_carlo_integration',
    'inverse_transform_sampling',
    'rejection_sampling',
    'importance_sampling',
    'metropolis_hastings',
    'gibbs_sampling',
    'slice_sampling',
    'hamiltonian_monte_carlo',
    'langevin_dynamics',
    'parallel_tempering',
    'adaptive_metropolis',
]
