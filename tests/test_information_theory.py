"""
Unit tests for information theory module
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import math
from information_theory import (
    entropy, cross_entropy, kl_divergence, js_divergence,
    mutual_information, information_gain,
    binary_cross_entropy, gini_impurity
)


class TestInformationTheory(unittest.TestCase):
    """Test information theory measures."""

    def test_entropy(self):
        """Test Shannon entropy."""
        # Fair coin: maximum entropy for 2 outcomes
        self.assertAlmostEqual(entropy([0.5, 0.5]), 1.0, places=5)
        # Certain outcome: zero entropy
        self.assertAlmostEqual(entropy([1.0, 0.0]), 0.0, places=5)
        # Uniform over 4 outcomes
        self.assertAlmostEqual(entropy([0.25, 0.25, 0.25, 0.25]), 2.0, places=5)

    def test_cross_entropy(self):
        """Test cross-entropy."""
        # Perfect prediction
        ce = cross_entropy([1.0, 0.0], [1.0, 0.0])
        self.assertAlmostEqual(ce, 0.0, places=5)

        # Uncertain prediction
        ce = cross_entropy([1.0, 0.0], [0.5, 0.5])
        self.assertAlmostEqual(ce, 1.0, places=5)

    def test_kl_divergence(self):
        """Test KL divergence."""
        # Identical distributions
        kl = kl_divergence([0.5, 0.5], [0.5, 0.5])
        self.assertAlmostEqual(kl, 0.0, places=5)

        # Different distributions
        kl = kl_divergence([0.9, 0.1], [0.5, 0.5])
        self.assertGreater(kl, 0.0)

    def test_js_divergence(self):
        """Test Jensen-Shannon divergence."""
        # Identical distributions
        js = js_divergence([0.5, 0.5], [0.5, 0.5])
        self.assertAlmostEqual(js, 0.0, places=5)

        # Completely different distributions
        js = js_divergence([1.0, 0.0], [0.0, 1.0])
        self.assertAlmostEqual(js, 1.0, places=5)

    def test_mutual_information(self):
        """Test mutual information."""
        # Independent variables: I(X;Y) = 0
        joint = [[0.25, 0.25], [0.25, 0.25]]
        mi = mutual_information(joint)
        self.assertAlmostEqual(mi, 0.0, places=5)

    def test_information_gain(self):
        """Test information gain for decision trees."""
        parent = [0.5, 0.5]
        children = [[0.8, 0.2], [0.2, 0.8]]
        split_prob = [0.5, 0.5]

        ig = information_gain(parent, children, split_prob)
        self.assertGreater(ig, 0.0)  # Should have positive information gain

    def test_binary_cross_entropy(self):
        """Test binary cross-entropy loss."""
        y_true = [1, 0, 1]
        y_pred = [0.9, 0.1, 0.8]
        bce = binary_cross_entropy(y_true, y_pred)
        self.assertGreater(bce, 0.0)

    def test_gini_impurity(self):
        """Test Gini impurity."""
        # Pure distribution
        self.assertAlmostEqual(gini_impurity([1.0, 0.0]), 0.0, places=5)
        # Maximum impurity for binary
        self.assertAlmostEqual(gini_impurity([0.5, 0.5]), 0.5, places=5)


if __name__ == '__main__':
    unittest.main()
