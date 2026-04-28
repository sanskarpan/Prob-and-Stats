"""
Probability Theory
==================

Basic probability concepts including conditional probability,
Bayes' theorem, and independence.

Applications in ML/DL:
- Bayesian inference for uncertainty quantification
- Conditional probability for graphical models
- Independence assumptions in naive Bayes classifiers
"""

from typing import Dict, List, Callable
import math


def conditional_probability(p_a_and_b: float, p_b: float) -> float:
    """
    Compute P(A|B) = P(A ∩ B) / P(B)

    Args:
        p_a_and_b: Probability of A and B occurring together
        p_b: Probability of B

    Returns:
        Conditional probability P(A|B)

    Example:
        >>> conditional_probability(0.3, 0.5)
        0.6
    """
    if p_b == 0:
        raise ValueError("P(B) cannot be zero")
    return p_a_and_b / p_b


def bayes_theorem(p_b_given_a: float, p_a: float, p_b: float) -> float:
    """
    Bayes' Theorem: P(A|B) = P(B|A) * P(A) / P(B)

    Critical for ML: Bayesian inference, spam classification, etc.

    Args:
        p_b_given_a: P(B|A) - likelihood
        p_a: P(A) - prior probability
        p_b: P(B) - evidence/marginal probability

    Returns:
        P(A|B) - posterior probability

    Example:
        Medical test: P(disease|positive test)
        >>> p_pos_given_disease = 0.99  # sensitivity
        >>> p_disease = 0.01  # prevalence
        >>> p_positive = 0.02  # test positive rate
        >>> bayes_theorem(p_pos_given_disease, p_disease, p_positive)
        0.495
    """
    if p_b == 0:
        raise ValueError("P(B) cannot be zero")
    return (p_b_given_a * p_a) / p_b


def bayes_theorem_with_partitions(
    p_b_given_a: float,
    p_a: float,
    partitions: List[float],
    likelihoods: List[float]
) -> float:
    """
    Bayes' Theorem using law of total probability for P(B).

    P(A|B) = P(B|A) * P(A) / [P(B|A₁)*P(A₁) + P(B|A₂)*P(A₂) + ...]

    Args:
        p_b_given_a: P(B|A) - likelihood for hypothesis A
        p_a: P(A) - prior for hypothesis A
        partitions: List of P(Aᵢ) for all hypotheses (must sum to 1)
        likelihoods: List of P(B|Aᵢ) for all hypotheses

    Returns:
        P(A|B) - posterior probability
    """
    if len(partitions) != len(likelihoods):
        raise ValueError("Partitions and likelihoods must have same length")

    if not math.isclose(sum(partitions), 1.0, rel_tol=1e-9):
        raise ValueError("Partitions must sum to 1")

    # Compute P(B) using law of total probability
    p_b = sum(lik * prob for lik, prob in zip(likelihoods, partitions))

    return bayes_theorem(p_b_given_a, p_a, p_b)


def law_of_total_probability(
    event_probs: List[float],
    conditional_probs: List[float]
) -> float:
    """
    Law of Total Probability: P(B) = Σ P(B|Aᵢ) * P(Aᵢ)

    Args:
        event_probs: List of P(Aᵢ) for partition events
        conditional_probs: List of P(B|Aᵢ) for each partition

    Returns:
        P(B) - total probability

    Example:
        >>> event_probs = [0.3, 0.7]  # P(A₁), P(A₂)
        >>> conditional_probs = [0.8, 0.4]  # P(B|A₁), P(B|A₂)
        >>> law_of_total_probability(event_probs, conditional_probs)
        0.52
    """
    if len(event_probs) != len(conditional_probs):
        raise ValueError("Lists must have same length")

    if not math.isclose(sum(event_probs), 1.0, rel_tol=1e-9):
        raise ValueError("Event probabilities must sum to 1")

    return sum(p_b_given_a * p_a
               for p_b_given_a, p_a in zip(conditional_probs, event_probs))


def are_independent(p_a: float, p_b: float, p_a_and_b: float,
                   tolerance: float = 1e-9) -> bool:
    """
    Check if two events A and B are independent.

    Events are independent if P(A ∩ B) = P(A) * P(B)

    Args:
        p_a: Probability of event A
        p_b: Probability of event B
        p_a_and_b: Probability of A and B together
        tolerance: Numerical tolerance for comparison

    Returns:
        True if events are independent

    Example:
        >>> are_independent(0.5, 0.5, 0.25)
        True
        >>> are_independent(0.5, 0.5, 0.3)
        False
    """
    expected = p_a * p_b
    return math.isclose(p_a_and_b, expected, rel_tol=tolerance)


def joint_probability_independent(probs: List[float]) -> float:
    """
    Compute joint probability for independent events.

    P(A₁ ∩ A₂ ∩ ... ∩ Aₙ) = P(A₁) * P(A₂) * ... * P(Aₙ)

    Args:
        probs: List of individual event probabilities

    Returns:
        Joint probability

    Example:
        >>> joint_probability_independent([0.5, 0.8, 0.9])
        0.36
    """
    result = 1.0
    for p in probs:
        if not 0 <= p <= 1:
            raise ValueError(f"Probability {p} not in [0, 1]")
        result *= p
    return result


def complement(p: float) -> float:
    """
    Compute complement probability: P(Ā) = 1 - P(A)

    Args:
        p: Probability of event A

    Returns:
        Probability of complement event

    Example:
        >>> complement(0.3)
        0.7
    """
    if not 0 <= p <= 1:
        raise ValueError(f"Probability {p} not in [0, 1]")
    return 1.0 - p


def union_probability(p_a: float, p_b: float, p_a_and_b: float) -> float:
    """
    Compute P(A ∪ B) = P(A) + P(B) - P(A ∩ B)

    Args:
        p_a: Probability of event A
        p_b: Probability of event B
        p_a_and_b: Probability of A and B together

    Returns:
        Probability of union

    Example:
        >>> union_probability(0.5, 0.4, 0.2)
        0.7
    """
    return p_a + p_b - p_a_and_b


def odds_to_probability(odds: float) -> float:
    """
    Convert odds to probability.

    P(A) = odds / (1 + odds)

    Args:
        odds: Odds in favor of event (e.g., 3:1 → 3.0)

    Returns:
        Probability

    Example:
        >>> odds_to_probability(3.0)  # 3:1 odds
        0.75
    """
    if odds < 0:
        raise ValueError("Odds must be non-negative")
    return odds / (1.0 + odds)


def probability_to_odds(p: float) -> float:
    """
    Convert probability to odds.

    odds = P(A) / P(Ā) = P(A) / (1 - P(A))

    Args:
        p: Probability of event

    Returns:
        Odds in favor

    Example:
        >>> probability_to_odds(0.75)
        3.0
    """
    if not 0 <= p < 1:
        raise ValueError(f"Probability {p} not in [0, 1)")
    return p / (1.0 - p)


def bayes_factor(p_data_given_h1: float, p_data_given_h0: float) -> float:
    """
    Compute Bayes factor for model comparison.

    BF = P(data|H₁) / P(data|H₀)

    Used in ML for model selection.

    Args:
        p_data_given_h1: Likelihood under hypothesis 1
        p_data_given_h0: Likelihood under hypothesis 0 (null)

    Returns:
        Bayes factor (BF > 1 favors H₁, BF < 1 favors H₀)

    Example:
        >>> bayes_factor(0.8, 0.2)
        4.0
    """
    if p_data_given_h0 == 0:
        raise ValueError("P(data|H₀) cannot be zero")
    return p_data_given_h1 / p_data_given_h0


def posterior_odds(prior_odds: float, bayes_factor: float) -> float:
    """
    Compute posterior odds from prior odds and Bayes factor.

    Posterior odds = Prior odds × Bayes factor

    Args:
        prior_odds: Prior odds in favor of hypothesis
        bayes_factor: Bayes factor from data

    Returns:
        Posterior odds

    Example:
        >>> posterior_odds(1.0, 4.0)  # Equal prior, BF=4
        4.0
    """
    return prior_odds * bayes_factor
