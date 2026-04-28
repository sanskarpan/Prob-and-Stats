"""
Probability Distributions and Statistics Tutorial
==================================================

Demonstrates probability distributions, sampling, and statistical analysis.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
random.seed(42)

from probability import bayes_theorem, conditional_probability
from distributions import Normal, Binomial, Bernoulli, Poisson
from statistics import mean, std, t_test, correlation, ab_test
from information_theory import entropy, kl_divergence


def bayesian_inference_example():
    """Demonstrate Bayesian inference with medical test."""
    print("=" * 70)
    print("BAYESIAN INFERENCE - Medical Test Example")
    print("=" * 70)

    print("\nScenario: Testing for a rare disease")
    print("-" * 40)

    # Prior probabilities
    p_disease = 0.01  # 1% of population has disease
    p_no_disease = 0.99

    # Test characteristics
    sensitivity = 0.99  # P(positive | disease)
    specificity = 0.95  # P(negative | no disease)
    p_pos_given_no_disease = 1 - specificity  # False positive rate

    print(f"Disease prevalence: {p_disease*100}%")
    print(f"Test sensitivity: {sensitivity*100}%")
    print(f"Test specificity: {specificity*100}%")

    # Compute P(positive) using law of total probability
    p_positive = sensitivity * p_disease + p_pos_given_no_disease * p_no_disease

    # Bayes' Theorem: P(disease | positive test)
    p_disease_given_pos = bayes_theorem(sensitivity, p_disease, p_positive)

    print(f"\nIf you test positive:")
    print(f"  P(have disease | positive) = {p_disease_given_pos:.4f} ({p_disease_given_pos*100:.2f}%)")
    print(f"\nInterpretation: Even with 99% sensitive test, positive result only")
    print(f"means ~{p_disease_given_pos*100:.1f}% chance of having disease due to low prevalence!")


def distribution_comparison():
    """Compare different probability distributions."""
    print("\n" + "=" * 70)
    print("PROBABILITY DISTRIBUTIONS - Properties and Sampling")
    print("=" * 70)

    print("\n1. Bernoulli Distribution (Coin Flip)")
    print("-" * 40)
    bern = Bernoulli(p=0.7)
    samples = bern.sample(1000)
    empirical_mean = sum(samples) / len(samples)
    print(f"   Parameter: p = 0.7")
    print(f"   Theoretical mean = {bern.mean():.4f}")
    print(f"   Empirical mean (1000 samples) = {empirical_mean:.4f}")
    print(f"   Theoretical variance = {bern.variance():.4f}")

    print("\n2. Normal Distribution")
    print("-" * 40)
    norm = Normal(mu=100, sigma=15)
    samples_norm = norm.sample(10000)
    print(f"   Parameters: μ=100, σ=15")
    print(f"   Theoretical mean = {norm.mean():.4f}")
    print(f"   Empirical mean = {mean(samples_norm):.4f}")
    print(f"   Theoretical std = {norm.std():.4f}")
    print(f"   Empirical std = {std(samples_norm, ddof=1):.4f}")
    print(f"   P(X ≤ 100) = {norm.cdf(100):.4f} (should be 0.5)")
    print(f"   P(X ≤ 115) = {norm.cdf(115):.4f} (68% rule: μ+σ)")

    print("\n3. Binomial Distribution")
    print("-" * 40)
    binom = Binomial(n=100, p=0.3)
    print(f"   Parameters: n=100, p=0.3")
    print(f"   Mean = {binom.mean():.4f}")
    print(f"   Std = {binom.std():.4f}")
    print(f"   P(X = 30) = {binom.pmf(30):.4f}")

    print("\n4. Poisson Distribution (Rare Events)")
    print("-" * 40)
    poisson = Poisson(lam=5.0)
    print(f"   Parameter: λ=5.0 (rate)")
    print(f"   Mean = Variance = {poisson.mean():.4f}")
    print(f"   P(X = 5) = {poisson.pmf(5):.4f}")


def hypothesis_testing_example():
    """Demonstrate hypothesis testing."""
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTING")
    print("=" * 70)

    print("\n1. One-Sample T-Test")
    print("-" * 40)
    print("   Question: Is the sample mean significantly different from 100?")

    # Generate sample from Normal(105, 15)
    norm = Normal(mu=105, sigma=15)
    sample = norm.sample(50)

    print(f"   Sample size: {len(sample)}")
    print(f"   Sample mean: {mean(sample):.4f}")
    print(f"   Hypothesized population mean: 100")

    t_stat, reject = t_test(sample, population_mean=100, alpha=0.05)
    print(f"   T-statistic: {t_stat:.4f}")
    print(f"   Reject null hypothesis: {reject}")
    print(f"   Conclusion: Sample mean {'is' if reject else 'is not'} significantly different from 100")

    print("\n2. A/B Testing")
    print("-" * 40)
    print("   Comparing conversion rates between control and treatment groups")

    # Simulate A/B test
    # Control: 10% conversion, Treatment: 15% conversion
    control_conversions = [1 if random.random() < 0.10 else 0 for _ in range(1000)]
    treatment_conversions = [1 if random.random() < 0.15 else 0 for _ in range(1000)]

    result = ab_test(control_conversions, treatment_conversions)

    print(f"   Control conversion rate: {result['control_rate']*100:.2f}%")
    print(f"   Treatment conversion rate: {result['treatment_rate']*100:.2f}%")
    print(f"   Lift: {result['lift']*100:.2f}%")
    print(f"   Z-statistic: {result['z_statistic']:.4f}")
    print(f"   P-value: {result['p_value']:.4f}")
    print(f"   Statistically significant: {result['significant']}")


def information_theory_example():
    """Demonstrate information theory concepts."""
    print("\n" + "=" * 70)
    print("INFORMATION THEORY")
    print("=" * 70)

    print("\n1. Entropy - Measuring Uncertainty")
    print("-" * 40)

    # Fair coin
    fair_coin = [0.5, 0.5]
    h_fair = entropy(fair_coin)
    print(f"   Fair coin [0.5, 0.5]: H = {h_fair:.4f} bits")

    # Biased coin
    biased_coin = [0.9, 0.1]
    h_biased = entropy(biased_coin)
    print(f"   Biased coin [0.9, 0.1]: H = {h_biased:.4f} bits")
    print(f"   → More certainty = Lower entropy")

    # Uniform distribution over 4 outcomes
    uniform_4 = [0.25, 0.25, 0.25, 0.25]
    h_uniform = entropy(uniform_4)
    print(f"   Uniform over 4 outcomes: H = {h_uniform:.4f} bits")

    print("\n2. KL Divergence - Measuring Distribution Difference")
    print("-" * 40)

    p = [0.7, 0.2, 0.1]
    q = [0.6, 0.3, 0.1]

    kl = kl_divergence(p, q)
    print(f"   P = {p}")
    print(f"   Q = {q}")
    print(f"   D_KL(P || Q) = {kl:.4f}")
    print(f"   → Non-zero KL divergence indicates distributions differ")


def central_limit_theorem_demo():
    """Demonstrate Central Limit Theorem."""
    print("\n" + "=" * 70)
    print("CENTRAL LIMIT THEOREM")
    print("=" * 70)

    print("\nSampling from Bernoulli(0.3) distribution")
    print("Taking sample means of size 30")
    print("-" * 40)

    bern = Bernoulli(p=0.3)
    sample_means = []

    for _ in range(1000):
        sample = bern.sample(30)
        sample_means.append(mean(sample))

    print(f"   Population mean: {bern.mean():.4f}")
    print(f"   Mean of sample means: {mean(sample_means):.4f}")
    print(f"   Population std: {bern.std():.4f}")
    print(f"   Std of sample means: {std(sample_means, ddof=1):.4f}")
    print(f"   Theoretical SE = σ/√n = {bern.std()/pow(30, 0.5):.4f}")
    print(f"\n   → Sample means are approximately normally distributed!")
    print(f"   → Standard error matches σ/√n prediction")


def main():
    """Run all tutorial examples."""
    bayesian_inference_example()
    distribution_comparison()
    hypothesis_testing_example()
    information_theory_example()
    central_limit_theorem_demo()

    print("\n" + "=" * 70)
    print("Tutorial Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("✓ Bayes' theorem updates beliefs with evidence")
    print("✓ Different distributions model different phenomena")
    print("✓ Hypothesis testing quantifies statistical significance")
    print("✓ Information theory measures uncertainty and divergence")
    print("✓ Central Limit Theorem: sample means → normal distribution")
    print("\nYou now understand core probability and statistics concepts!")


if __name__ == '__main__':
    main()
