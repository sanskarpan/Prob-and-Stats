"""
Unit tests for statistics module
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from statistics import (
    mean, median, mode, variance, std,
    covariance, correlation,
    z_score, t_test, two_sample_t_test,
    z_test, ab_test
)


class TestDescriptiveStatistics(unittest.TestCase):
    """Test descriptive statistics."""

    def test_mean(self):
        """Test arithmetic mean."""
        data = [1, 2, 3, 4, 5]
        self.assertAlmostEqual(mean(data), 3.0, places=5)

    def test_median(self):
        """Test median."""
        # Odd length
        self.assertAlmostEqual(median([1, 2, 3, 4, 5]), 3.0, places=5)
        # Even length
        self.assertAlmostEqual(median([1, 2, 3, 4]), 2.5, places=5)

    def test_mode(self):
        """Test mode."""
        self.assertEqual(mode([1, 2, 2, 3, 4]), 2)

    def test_variance(self):
        """Test variance."""
        data = [1, 2, 3, 4, 5]
        # Population variance
        pop_var = variance(data, ddof=0)
        self.assertAlmostEqual(pop_var, 2.0, places=5)
        # Sample variance
        sample_var = variance(data, ddof=1)
        self.assertAlmostEqual(sample_var, 2.5, places=5)

    def test_std(self):
        """Test standard deviation."""
        data = [1, 2, 3, 4, 5]
        std_val = std(data, ddof=1)
        self.assertAlmostEqual(std_val, 1.5811, places=3)

    def test_covariance(self):
        """Test covariance."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        cov = covariance(x, y, ddof=0)
        self.assertAlmostEqual(cov, 4.0, places=5)

    def test_correlation(self):
        """Test Pearson correlation."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        corr = correlation(x, y)
        self.assertAlmostEqual(corr, 1.0, places=5)


class TestHypothesisTesting(unittest.TestCase):
    """Test hypothesis testing."""

    def test_z_score(self):
        """Test z-score computation."""
        data = [1, 2, 3, 4, 5]
        z = z_score(data, 5)
        self.assertGreater(z, 1.0)  # 5 is above mean

    def test_t_test(self):
        """Test one-sample t-test."""
        # Sample clearly different from hypothesized mean
        sample = [1, 2, 3, 4, 5]
        t_stat, reject = t_test(sample, 10, alpha=0.05)
        self.assertTrue(reject)

    def test_two_sample_t_test(self):
        """Test two-sample t-test."""
        sample1 = [1, 2, 3, 4, 5]
        sample2 = [6, 7, 8, 9, 10]
        t_stat, reject = two_sample_t_test(sample1, sample2, alpha=0.05)
        self.assertTrue(reject)  # Means are clearly different

    def test_z_test(self):
        """Test z-test with known population std."""
        # Large sample clearly different from mu=10 → reject
        sample = [1, 2, 3, 4, 5]
        z_stat, reject = z_test(sample, population_mean=10, population_std=2.0)
        self.assertTrue(reject)
        # Sample consistent with mu=3 → do not reject
        _, reject2 = z_test([3.0, 3.0, 3.0, 3.0, 3.0], population_mean=3.0, population_std=1.0)
        self.assertFalse(reject2)
        # Non-standard alpha
        z_stat3, reject3 = z_test([1, 2, 3, 4, 5], population_mean=10, population_std=2.0, alpha=0.01)
        self.assertTrue(reject3)

    def test_ab_test(self):
        """Test A/B test."""
        control = [0, 0, 1, 0, 1, 1, 0, 0, 1, 0]
        treatment = [1, 1, 1, 0, 1, 1, 1, 0, 1, 1]
        result = ab_test(control, treatment)

        self.assertIn('control_rate', result)
        self.assertIn('treatment_rate', result)
        self.assertIn('z_statistic', result)
        self.assertIn('significant', result)


if __name__ == '__main__':
    unittest.main()
