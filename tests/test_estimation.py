"""
Unit tests for estimation module — MLE, MAP, Bayesian inference.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import math
import random

from estimation import (
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


class TestMaximumLikelihood(unittest.TestCase):
    """Tests for MLE closed-form estimates."""

    def test_normal_mle(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        mle = maximum_likelihood(data, 'normal')
        self.assertAlmostEqual(mle['mu'], 3.0, places=9)
        self.assertAlmostEqual(mle['sigma'], math.sqrt(2.0), places=9)

    def test_bernoulli_mle(self):
        data = [1.0, 0.0, 1.0, 1.0, 0.0]
        mle = maximum_likelihood(data, 'bernoulli')
        self.assertAlmostEqual(mle['p'], 0.6, places=9)

    def test_poisson_mle(self):
        data = [2.0, 3.0, 5.0, 1.0, 4.0]
        mle = maximum_likelihood(data, 'poisson')
        self.assertAlmostEqual(mle['lambda'], 3.0, places=9)

    def test_exponential_mle(self):
        # MLE for Exponential: lambda = 1 / x_bar
        data = [0.5, 1.0, 1.5, 2.0]
        mle = maximum_likelihood(data, 'exponential')
        expected_lambda = 1.0 / (sum(data) / len(data))
        self.assertAlmostEqual(mle['lambda'], expected_lambda, places=9)

    def test_unknown_distribution_raises(self):
        with self.assertRaises(ValueError):
            maximum_likelihood([1.0, 2.0], 'unknown')


class TestMaximumAPosteriori(unittest.TestCase):
    """Tests for MAP estimates."""

    def test_bernoulli_map_uniform_prior_equals_mle(self):
        # Beta(1,1) prior is uniform; MAP should match MLE
        data = [1.0, 0.0, 1.0, 1.0, 0.0]
        prior = {'alpha': 1.0, 'beta': 1.0}
        map_est = maximum_a_posteriori(data, prior, 'bernoulli')
        self.assertAlmostEqual(map_est['p'], 0.6, places=9)

    def test_bernoulli_map_informative_prior(self):
        # alpha=5, beta=5 (strong prior at 0.5); data: 9 successes out of 10
        # MAP = (5+9-1) / (5+5+10-2) = 13/18 ≈ 0.722
        data = [1.0] * 9 + [0.0]
        prior = {'alpha': 5.0, 'beta': 5.0}
        map_est = maximum_a_posteriori(data, prior, 'bernoulli')
        expected = (5 + 9 - 1) / (5 + 5 + 10 - 2)
        self.assertAlmostEqual(map_est['p'], expected, places=9)

    def test_normal_map_shrinks_to_prior(self):
        # With very strong prior (high precision), MAP should be close to prior mean
        data = [5.0]
        prior = {'mu_prior': 0.0, 'precision': 1000.0, 'sigma': 1.0}
        map_est = maximum_a_posteriori(data, prior, 'normal')
        # Posterior mu ≈ 0.0 (prior dominates)
        self.assertLess(abs(map_est['mu']), 0.1)

    def test_unknown_distribution_raises(self):
        with self.assertRaises(ValueError):
            maximum_a_posteriori([1.0], {}, 'unknown')


class TestBayesianUpdate(unittest.TestCase):
    """Tests for conjugate Bayesian updates."""

    def test_normal_normal_conjugate(self):
        # Prior: mu ~ N(0, 1/1=1), sigma=1, precision=1
        # Likelihood: X ~ N(mu, 1), n=3, x_bar=2
        # Posterior tau_n = 1 + 3*1 = 4, mu_n = (1*0 + 3*1*2)/4 = 1.5, sigma_n = 0.5
        data = [1.0, 2.0, 3.0]
        prior = {'mu': 0.0, 'sigma': 1.0, 'precision': 1.0}
        post = bayesian_update(prior, data, 'normal')
        self.assertAlmostEqual(post['mu'], 1.5, places=9)
        self.assertAlmostEqual(post['sigma'], 0.5, places=9)
        self.assertAlmostEqual(post['precision'], 4.0, places=9)

    def test_beta_bernoulli_conjugate(self):
        # Prior: p ~ Beta(2, 3); data: 7 successes, 3 failures
        # Posterior: Beta(9, 6)
        prior = {'alpha': 2.0, 'beta': 3.0}
        data = [1.0] * 7 + [0.0] * 3
        post = bayesian_update(prior, data, 'bernoulli')
        self.assertAlmostEqual(post['alpha'], 9.0, places=9)
        self.assertAlmostEqual(post['beta'], 6.0, places=9)
        self.assertAlmostEqual(post['mean'], 9.0 / 15.0, places=9)

    def test_gamma_poisson_conjugate(self):
        # Prior: lambda ~ Gamma(2, 1); data: [3, 5, 2] sum=10, n=3
        # Posterior: Gamma(2+10, 1+3) = Gamma(12, 4), mean=3
        prior = {'alpha': 2.0, 'beta': 1.0}
        data = [3.0, 5.0, 2.0]
        post = bayesian_update(prior, data, 'poisson')
        self.assertAlmostEqual(post['alpha'], 12.0, places=9)
        self.assertAlmostEqual(post['beta'], 4.0, places=9)
        self.assertAlmostEqual(post['mean'], 3.0, places=9)

    def test_unknown_distribution_raises(self):
        with self.assertRaises(ValueError):
            bayesian_update({}, [1.0], 'unknown')


class TestCredibleInterval(unittest.TestCase):
    """Tests for Bayesian credible intervals."""

    def test_normal_credible_interval_symmetric(self):
        # N(0, 1): 95% CI should be approximately [-1.96, +1.96]
        post = {'mu': 0.0, 'sigma': 1.0}
        lo, hi = credible_interval(post, 'normal', 0.95)
        self.assertAlmostEqual(lo, -1.96, places=1)
        self.assertAlmostEqual(hi, +1.96, places=1)
        self.assertAlmostEqual(lo + hi, 0.0, places=6)  # symmetric

    def test_normal_credible_interval_shifted(self):
        # N(5, 0.5): 90% CI
        post = {'mu': 5.0, 'sigma': 0.5}
        lo, hi = credible_interval(post, 'normal', 0.90)
        self.assertGreater(lo, 4.0)
        self.assertLess(hi, 6.0)
        self.assertAlmostEqual((lo + hi) / 2, 5.0, places=6)

    def test_beta_credible_interval_symmetric(self):
        # Beta(50, 50): 95% CI should be close to [0.43, 0.57]
        post = {'alpha': 50.0, 'beta': 50.0}
        lo, hi = credible_interval(post, 'bernoulli', 0.95)
        self.assertGreater(lo, 0.4)
        self.assertLess(hi, 0.6)
        self.assertTrue(lo < 0.5 < hi)

    def test_beta_credible_interval_exact_for_skewed(self):
        # Beta(1, 50): strongly left-skewed, mean ≈ 0.02
        # Exact 95% CI should have upper bound < 0.1
        post = {'alpha': 1.0, 'beta': 50.0}
        lo, hi = credible_interval(post, 'bernoulli', 0.95)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 0.1)  # Normal approx would give wrong wide interval

    def test_beta_credible_interval_bounds(self):
        # CI must lie in [0, 1]
        for a, b in [(0.5, 0.5), (1, 1), (2, 10), (100, 1)]:
            post = {'alpha': a, 'beta': b}
            lo, hi = credible_interval(post, 'bernoulli', 0.95)
            self.assertGreaterEqual(lo, 0.0)
            self.assertLessEqual(hi, 1.0)
            self.assertLessEqual(lo, hi)

    def test_unknown_distribution_raises(self):
        with self.assertRaises(ValueError):
            credible_interval({}, 'unknown', 0.95)


class TestPosteriorPredictive(unittest.TestCase):
    """Tests for posterior predictive samples."""

    def test_normal_posterior_predictive_shape(self):
        random.seed(42)
        post = {'mu': 3.0, 'sigma': 1.0}
        samples = posterior_predictive(post, 'normal', n_samples=1000)
        self.assertEqual(len(samples), 1000)
        emp_mean = sum(samples) / len(samples)
        self.assertAlmostEqual(emp_mean, 3.0, delta=0.2)

    def test_bernoulli_posterior_predictive_binary(self):
        random.seed(42)
        post = {'alpha': 8.0, 'beta': 2.0}
        samples = posterior_predictive(post, 'bernoulli', n_samples=500)
        self.assertTrue(all(s in (0, 1) for s in samples))
        emp_rate = sum(samples) / len(samples)
        # Posterior mean p = 8/10 = 0.8
        self.assertAlmostEqual(emp_rate, 0.8, delta=0.1)


class TestEmpiricalBayes(unittest.TestCase):
    """Tests for empirical Bayes prior estimation."""

    def test_empirical_bayes_normal(self):
        groups = [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]
        prior = empirical_bayes(groups, 'normal')
        self.assertIn('mu', prior)
        self.assertIn('sigma', prior)
        # Group means: [2, 3, 4]; prior mu should be 3.0
        self.assertAlmostEqual(prior['mu'], 3.0, places=9)


class TestLikelihoodRatio(unittest.TestCase):
    """Tests for likelihood ratio computation."""

    def test_likelihood_ratio_same_model_is_one(self):
        data = [1.0, 2.0, 3.0]
        params = {'mu': 2.0, 'sigma': 1.0}
        lr = likelihood_ratio(data, params, params, 'normal')
        self.assertAlmostEqual(lr, 1.0, places=6)

    def test_likelihood_ratio_better_model_greater_than_one(self):
        # Data from N(2, 1); model1 closer to truth
        data = [1.8, 2.1, 1.9, 2.2, 1.7]
        params1 = {'mu': 2.0, 'sigma': 1.0}
        params2 = {'mu': 10.0, 'sigma': 1.0}
        lr = likelihood_ratio(data, params1, params2, 'normal')
        self.assertGreater(lr, 1.0)


class TestBootstrapParameterEstimate(unittest.TestCase):
    """Tests for bootstrap parameter estimation."""

    def test_bootstrap_mean_estimate(self):
        random.seed(42)
        from statistics import mean
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        est, se, ci = bootstrap_parameter_estimate(data, mean, n_bootstrap=500)
        self.assertAlmostEqual(est, 3.0, places=9)
        self.assertGreater(se, 0.0)
        self.assertLess(ci[0], 3.0)
        self.assertGreater(ci[1], 3.0)

    def test_bootstrap_ci_coverage(self):
        # CI should contain the estimate
        random.seed(0)
        from statistics import mean
        data = list(range(1, 11))
        est, se, ci = bootstrap_parameter_estimate(data, mean, n_bootstrap=200)
        self.assertGreaterEqual(est, ci[0])
        self.assertLessEqual(est, ci[1])


class TestConjugatePriorParameters(unittest.TestCase):
    """Tests for default conjugate prior parameters."""

    def test_bernoulli_uniform_prior(self):
        prior = conjugate_prior_parameters('bernoulli')
        self.assertEqual(prior['alpha'], 1.0)
        self.assertEqual(prior['beta'], 1.0)

    def test_poisson_prior(self):
        prior = conjugate_prior_parameters('poisson')
        self.assertIn('alpha', prior)
        self.assertIn('beta', prior)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            conjugate_prior_parameters('unknown')


if __name__ == '__main__':
    unittest.main()
