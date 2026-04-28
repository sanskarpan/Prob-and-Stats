"""
Unit tests for distributions module
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import math
import random
from distributions import (
    Bernoulli, Binomial, Poisson, Categorical,
    Uniform, Normal, Exponential, Beta, Gamma
)


class TestDiscreteDistributions(unittest.TestCase):
    """Test discrete probability distributions."""

    def test_bernoulli(self):
        """Test Bernoulli distribution."""
        bern = Bernoulli(0.7)

        # Test PMF
        self.assertAlmostEqual(bern.pmf(1), 0.7, places=5)
        self.assertAlmostEqual(bern.pmf(0), 0.3, places=5)

        # Test mean and variance
        self.assertAlmostEqual(bern.mean(), 0.7, places=5)
        self.assertAlmostEqual(bern.variance(), 0.21, places=5)

        # Test sampling
        random.seed(42)
        sample = bern.sample(100)
        self.assertIsInstance(sample, list)
        self.assertEqual(len(sample), 100)

    def test_binomial(self):
        """Test Binomial distribution."""
        binom = Binomial(n=10, p=0.5)

        # Test PMF: P(X=5) for n=10, p=0.5
        pmf_5 = binom.pmf(5)
        expected = 0.246  # Approximately
        self.assertAlmostEqual(pmf_5, expected, places=2)

        # Test mean and variance
        self.assertAlmostEqual(binom.mean(), 5.0, places=5)
        self.assertAlmostEqual(binom.variance(), 2.5, places=5)

    def test_poisson(self):
        """Test Poisson distribution."""
        poisson = Poisson(lam=3.0)

        # Test mean and variance
        self.assertAlmostEqual(poisson.mean(), 3.0, places=5)
        self.assertAlmostEqual(poisson.variance(), 3.0, places=5)

        # Test PMF is positive
        self.assertGreater(poisson.pmf(3), 0)

    def test_categorical(self):
        """Test Categorical distribution."""
        cat = Categorical([0.2, 0.3, 0.5])

        # Test PMF
        self.assertAlmostEqual(cat.pmf(0), 0.2, places=5)
        self.assertAlmostEqual(cat.pmf(2), 0.5, places=5)

        # Test sampling
        random.seed(42)
        sample = cat.sample(100)
        self.assertTrue(all(0 <= s < 3 for s in sample))


class TestContinuousDistributions(unittest.TestCase):
    """Test continuous probability distributions."""

    def test_uniform(self):
        """Test Uniform distribution."""
        unif = Uniform(a=0.0, b=1.0)

        # Test PDF
        self.assertAlmostEqual(unif.pdf(0.5), 1.0, places=5)
        self.assertAlmostEqual(unif.pdf(1.5), 0.0, places=5)

        # Test mean and variance
        self.assertAlmostEqual(unif.mean(), 0.5, places=5)
        self.assertAlmostEqual(unif.variance(), 1.0/12.0, places=5)

    def test_normal(self):
        """Test Normal distribution."""
        norm = Normal(mu=0.0, sigma=1.0)

        # Test PDF at mean
        pdf_0 = norm.pdf(0.0)
        expected = 1.0 / math.sqrt(2 * math.pi)
        self.assertAlmostEqual(pdf_0, expected, places=5)

        # Test mean and variance
        self.assertAlmostEqual(norm.mean(), 0.0, places=5)
        self.assertAlmostEqual(norm.variance(), 1.0, places=5)

        # Test CDF at mean (should be 0.5)
        self.assertAlmostEqual(norm.cdf(0.0), 0.5, places=2)

    def test_exponential(self):
        """Test Exponential distribution."""
        exp = Exponential(lam=1.0)

        # Test mean and variance
        self.assertAlmostEqual(exp.mean(), 1.0, places=5)
        self.assertAlmostEqual(exp.variance(), 1.0, places=5)

        # Test CDF
        self.assertAlmostEqual(exp.cdf(0.0), 0.0, places=5)

    def test_beta(self):
        """Test Beta distribution."""
        beta = Beta(alpha=2.0, beta=2.0)

        # Test mean for symmetric Beta
        self.assertAlmostEqual(beta.mean(), 0.5, places=5)

    def test_gamma(self):
        """Test Gamma distribution."""
        gamma = Gamma(alpha=2.0, beta=1.0)

        # Test mean
        self.assertAlmostEqual(gamma.mean(), 2.0, places=5)
        # Test variance
        self.assertAlmostEqual(gamma.variance(), 2.0, places=5)


if __name__ == '__main__':
    unittest.main()
