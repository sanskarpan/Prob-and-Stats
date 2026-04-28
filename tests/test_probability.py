"""
Unit tests for probability module
"""

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import math
from probability import (
    conditional_probability,
    bayes_theorem,
    law_of_total_probability,
    are_independent,
    joint_probability_independent,
    complement,
    union_probability,
    odds_to_probability,
    probability_to_odds,
    bayes_factor
)


class TestBasicProbability(unittest.TestCase):
    """Test basic probability operations."""

    def test_conditional_probability(self):
        """Test conditional probability P(A|B) = P(A∩B) / P(B)"""
        p_ab = 0.3
        p_b = 0.5
        result = conditional_probability(p_ab, p_b)
        self.assertAlmostEqual(result, 0.6, places=5)

    def test_conditional_probability_zero_denominator(self):
        """Test that division by zero raises error."""
        with self.assertRaises(ValueError):
            conditional_probability(0.3, 0.0)

    def test_bayes_theorem(self):
        """Test Bayes' theorem."""
        # Medical test example
        p_pos_given_disease = 0.99  # Sensitivity
        p_disease = 0.01  # Prevalence
        p_positive = 0.02  # Marginal positive rate

        result = bayes_theorem(p_pos_given_disease, p_disease, p_positive)
        expected = (0.99 * 0.01) / 0.02
        self.assertAlmostEqual(result, expected, places=5)

    def test_law_of_total_probability(self):
        """Test law of total probability."""
        event_probs = [0.3, 0.7]
        conditional_probs = [0.8, 0.4]
        result = law_of_total_probability(event_probs, conditional_probs)
        expected = 0.3 * 0.8 + 0.7 * 0.4
        self.assertAlmostEqual(result, expected, places=5)

    def test_independence(self):
        """Test independence check."""
        # Independent events
        self.assertTrue(are_independent(0.5, 0.5, 0.25))
        # Dependent events
        self.assertFalse(are_independent(0.5, 0.5, 0.3))

    def test_joint_probability_independent(self):
        """Test joint probability for independent events."""
        probs = [0.5, 0.8, 0.9]
        result = joint_probability_independent(probs)
        expected = 0.5 * 0.8 * 0.9
        self.assertAlmostEqual(result, expected, places=5)

    def test_complement(self):
        """Test complement probability."""
        self.assertAlmostEqual(complement(0.3), 0.7, places=5)
        self.assertAlmostEqual(complement(0.0), 1.0, places=5)
        self.assertAlmostEqual(complement(1.0), 0.0, places=5)

    def test_union_probability(self):
        """Test P(A∪B) = P(A) + P(B) - P(A∩B)"""
        result = union_probability(0.5, 0.4, 0.2)
        self.assertAlmostEqual(result, 0.7, places=5)

    def test_odds_to_probability(self):
        """Test odds to probability conversion."""
        # 3:1 odds → 0.75 probability
        self.assertAlmostEqual(odds_to_probability(3.0), 0.75, places=5)
        # 1:1 odds → 0.5 probability
        self.assertAlmostEqual(odds_to_probability(1.0), 0.5, places=5)

    def test_probability_to_odds(self):
        """Test probability to odds conversion."""
        self.assertAlmostEqual(probability_to_odds(0.75), 3.0, places=5)
        self.assertAlmostEqual(probability_to_odds(0.5), 1.0, places=5)

    def test_bayes_factor(self):
        """Test Bayes factor calculation."""
        bf = bayes_factor(0.8, 0.2)
        self.assertAlmostEqual(bf, 4.0, places=5)


if __name__ == '__main__':
    unittest.main()
