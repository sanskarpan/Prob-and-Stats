"""
Information Theory
==================

Entropy, divergence, and information measures.

Applications in ML/DL:
- Loss functions (cross-entropy loss)
- Decision trees (information gain)
- Model uncertainty quantification
- Variational inference
- GANs and generative models
"""

import math
from typing import List, Dict, Union


def entropy(probabilities: List[float], base: float = 2.0) -> float:
    """
    Compute Shannon entropy H(X) = -Σ p(x) log p(x)

    Measures average uncertainty/information content.

    Args:
        probabilities: Probability distribution (must sum to 1)
        base: Logarithm base (2 for bits, e for nats)

    Returns:
        Entropy value

    Example:
        >>> entropy([0.5, 0.5])  # Fair coin: maximum entropy
        1.0
        >>> entropy([1.0, 0.0])  # Certain outcome: zero entropy
        0.0
        >>> entropy([0.25, 0.25, 0.25, 0.25])  # Uniform over 4 outcomes
        2.0
    """
    if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-9):
        raise ValueError("Probabilities must sum to 1")

    if any(p < 0 for p in probabilities):
        raise ValueError("Probabilities must be non-negative")

    # H(X) = -Σ p(x) log p(x)
    # Note: 0 * log(0) = 0 by convention
    h = 0.0
    for p in probabilities:
        if p > 0:  # Skip zero probabilities
            h -= p * math.log(p) / math.log(base)

    return h


def joint_entropy(joint_probs: List[List[float]], base: float = 2.0) -> float:
    """
    Compute joint entropy H(X, Y) = -Σ Σ p(x,y) log p(x,y)

    Args:
        joint_probs: Joint probability distribution p(x,y)
        base: Logarithm base

    Returns:
        Joint entropy

    Example:
        >>> # Independent variables: H(X,Y) = H(X) + H(Y)
        >>> joint = [[0.25, 0.25], [0.25, 0.25]]
        >>> joint_entropy(joint)
        2.0
    """
    # Flatten to list of probabilities
    flat_probs = [p for row in joint_probs for p in row]
    return entropy(flat_probs, base)


def conditional_entropy(
    joint_probs: List[List[float]],
    base: float = 2.0
) -> float:
    """
    Compute conditional entropy H(Y|X) = H(X,Y) - H(X)

    Measures average uncertainty in Y given X.

    Args:
        joint_probs: Joint probability distribution p(x,y)
        base: Logarithm base

    Returns:
        Conditional entropy H(Y|X)

    Example:
        >>> # If Y is independent of X: H(Y|X) = H(Y)
        >>> joint = [[0.25, 0.25], [0.25, 0.25]]
        >>> conditional_entropy(joint)
        1.0
    """
    # H(Y|X) = H(X,Y) - H(X)
    h_xy = joint_entropy(joint_probs, base)

    # Compute H(X) - marginal entropy
    marginal_x = [sum(row) for row in joint_probs]
    h_x = entropy(marginal_x, base)

    return h_xy - h_x


def cross_entropy(
    p: List[float],
    q: List[float],
    base: float = 2.0
) -> float:
    """
    Compute cross-entropy H(p, q) = -Σ p(x) log q(x)

    Critical for ML: Used as loss function in classification.

    Args:
        p: True probability distribution
        q: Predicted probability distribution
        base: Logarithm base

    Returns:
        Cross-entropy value

    Example:
        >>> # Perfect prediction
        >>> cross_entropy([1.0, 0.0], [1.0, 0.0])
        0.0
        >>> # Uncertain prediction
        >>> cross_entropy([1.0, 0.0], [0.5, 0.5])
        1.0
    """
    if len(p) != len(q):
        raise ValueError("Distributions must have same length")

    if not math.isclose(sum(p), 1.0, rel_tol=1e-9):
        raise ValueError("p must sum to 1")

    if not math.isclose(sum(q), 1.0, rel_tol=1e-9):
        raise ValueError("q must sum to 1")

    if any(p_i < 0 for p_i in p):
        raise ValueError("p must be non-negative")

    if any(q_i < 0 for q_i in q):
        raise ValueError("q must be non-negative")

    # H(p, q) = -Σ p(x) log q(x)
    ce = 0.0
    for p_i, q_i in zip(p, q):
        if p_i > 0:
            if q_i <= 0:
                return float('inf')  # Infinite cross-entropy
            ce -= p_i * math.log(q_i) / math.log(base)

    return ce


def kl_divergence(
    p: List[float],
    q: List[float],
    base: float = 2.0
) -> float:
    """
    Compute Kullback-Leibler divergence D_KL(p || q) = Σ p(x) log(p(x)/q(x))

    Measures how one distribution differs from another.
    NOT symmetric: D_KL(p||q) ≠ D_KL(q||p)

    Used in: Variational inference, GANs, model comparison

    Args:
        p: First distribution (usually true distribution)
        q: Second distribution (usually approximation)
        base: Logarithm base

    Returns:
        KL divergence (always ≥ 0, = 0 iff p = q)

    Example:
        >>> # Identical distributions
        >>> kl_divergence([0.5, 0.5], [0.5, 0.5])
        0.0
        >>> # Different distributions
        >>> kl_divergence([0.9, 0.1], [0.5, 0.5])
        0.469...
    """
    if len(p) != len(q):
        raise ValueError("Distributions must have same length")

    if any(p_i < 0 for p_i in p):
        raise ValueError("p must be non-negative")

    if any(q_i < 0 for q_i in q):
        raise ValueError("q must be non-negative")

    # D_KL(p || q) = Σ p(x) log(p(x)/q(x))
    #              = H(p, q) - H(p)
    #              = cross_entropy(p, q) - entropy(p)
    kl = 0.0
    for p_i, q_i in zip(p, q):
        if p_i > 0:
            if q_i <= 0:
                return float('inf')
            kl += p_i * math.log(p_i / q_i) / math.log(base)

    return max(0.0, kl)  # Ensure non-negative (numerical stability)


def js_divergence(
    p: List[float],
    q: List[float],
    base: float = 2.0
) -> float:
    """
    Compute Jensen-Shannon divergence - symmetric version of KL divergence.

    JS(p, q) = 0.5 * D_KL(p || m) + 0.5 * D_KL(q || m)
    where m = 0.5 * (p + q)

    Args:
        p: First distribution
        q: Second distribution
        base: Logarithm base

    Returns:
        JS divergence (symmetric, bounded: 0 ≤ JS ≤ 1 for base=2)

    Example:
        >>> js_divergence([0.5, 0.5], [0.5, 0.5])
        0.0
        >>> js_divergence([1.0, 0.0], [0.0, 1.0])
        1.0
    """
    if len(p) != len(q):
        raise ValueError("Distributions must have same length")

    # Mixture distribution
    m = [(p_i + q_i) / 2 for p_i, q_i in zip(p, q)]

    # JS(p, q) = 0.5 * D_KL(p||m) + 0.5 * D_KL(q||m)
    return 0.5 * kl_divergence(p, m, base) + 0.5 * kl_divergence(q, m, base)


def mutual_information(
    joint_probs: List[List[float]],
    base: float = 2.0
) -> float:
    """
    Compute mutual information I(X; Y) = H(X) + H(Y) - H(X,Y)

    Measures how much knowing one variable reduces uncertainty about the other.
    Always ≥ 0, = 0 iff X and Y are independent.

    Used in: Feature selection, dependency detection

    Args:
        joint_probs: Joint probability distribution p(x,y)
        base: Logarithm base

    Returns:
        Mutual information

    Example:
        >>> # Independent variables: I(X;Y) = 0
        >>> joint = [[0.25, 0.25], [0.25, 0.25]]
        >>> mutual_information(joint)
        0.0
    """
    # Marginal distributions
    marginal_x = [sum(row) for row in joint_probs]
    marginal_y = [sum(col) for col in zip(*joint_probs)]

    # I(X; Y) = H(X) + H(Y) - H(X,Y)
    h_x = entropy(marginal_x, base)
    h_y = entropy(marginal_y, base)
    h_xy = joint_entropy(joint_probs, base)

    mi = h_x + h_y - h_xy
    return max(0.0, mi)  # Ensure non-negative


def information_gain(
    parent_probs: List[float],
    child_probs_list: List[List[float]],
    split_probs: List[float],
    base: float = 2.0
) -> float:
    """
    Compute information gain (used in decision trees).

    IG = H(parent) - Σ p(child) * H(child)

    Args:
        parent_probs: Probability distribution before split
        child_probs_list: List of probability distributions after split
        split_probs: Probability of each child
        base: Logarithm base

    Returns:
        Information gain

    Example:
        >>> parent = [0.5, 0.5]
        >>> children = [[0.8, 0.2], [0.2, 0.8]]
        >>> split_prob = [0.5, 0.5]
        >>> information_gain(parent, children, split_prob)
        0.278...
    """
    if not math.isclose(sum(split_probs), 1.0, rel_tol=1e-9):
        raise ValueError("Split probabilities must sum to 1")

    # H(parent)
    h_parent = entropy(parent_probs, base)

    # Weighted average of child entropies
    h_children = 0.0
    for child_probs, split_prob in zip(child_probs_list, split_probs):
        h_children += split_prob * entropy(child_probs, base)

    return h_parent - h_children


def perplexity(probabilities: List[float], base: float = 2.0) -> float:
    """
    Compute perplexity = base^H(X)

    Used to evaluate language models. Lower perplexity = better model.

    Args:
        probabilities: Probability distribution
        base: Logarithm base (typically 2 or e)

    Returns:
        Perplexity

    Example:
        >>> perplexity([0.5, 0.5])  # Uniform over 2: perplexity = 2
        2.0
        >>> perplexity([0.25, 0.25, 0.25, 0.25])  # Uniform over 4: perplexity = 4
        4.0
    """
    h = entropy(probabilities, base)
    return base ** h


def binary_cross_entropy(y_true: List[float], y_pred: List[float]) -> float:
    """
    Compute binary cross-entropy loss.

    BCE = -Σ [y * log(ŷ) + (1-y) * log(1-ŷ)]

    Used in binary classification.

    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted probabilities [0, 1]

    Returns:
        Binary cross-entropy

    Example:
        >>> binary_cross_entropy([1, 0, 1], [0.9, 0.1, 0.8])
        0.156...
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    eps = 1e-15  # Small epsilon to avoid log(0)
    bce = 0.0

    for y_t, y_p in zip(y_true, y_pred):
        # Clip predictions to avoid log(0)
        y_p = max(eps, min(1 - eps, y_p))

        bce -= y_t * math.log(y_p) + (1 - y_t) * math.log(1 - y_p)

    return bce / len(y_true)


def categorical_cross_entropy(
    y_true: List[List[float]],
    y_pred: List[List[float]]
) -> float:
    """
    Compute categorical cross-entropy loss.

    CCE = -Σ Σ y_true[i][j] * log(y_pred[i][j])

    Used in multi-class classification.

    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted probabilities

    Returns:
        Categorical cross-entropy

    Example:
        >>> y_true = [[1, 0, 0], [0, 1, 0]]
        >>> y_pred = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]
        >>> categorical_cross_entropy(y_true, y_pred)
        0.212...
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    eps = 1e-15
    cce = 0.0

    for true_dist, pred_dist in zip(y_true, y_pred):
        if len(true_dist) != len(pred_dist):
            raise ValueError("Distribution dimensions must match")

        for y_t, y_p in zip(true_dist, pred_dist):
            if y_t > 0:
                y_p = max(eps, y_p)  # Clip to avoid log(0)
                cce -= y_t * math.log(y_p)

    return cce / len(y_true)


def gini_impurity(probabilities: List[float]) -> float:
    """
    Compute Gini impurity (alternative to entropy for decision trees).

    Gini = 1 - Σ p(x)²

    Args:
        probabilities: Probability distribution

    Returns:
        Gini impurity (0 for pure, 0.5 for binary uniform)

    Example:
        >>> gini_impurity([1.0, 0.0])  # Pure
        0.0
        >>> gini_impurity([0.5, 0.5])  # Maximum impurity for binary
        0.5
    """
    if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-9):
        raise ValueError("Probabilities must sum to 1")

    return 1.0 - sum(p**2 for p in probabilities)


def relative_entropy(p: List[float], q: List[float]) -> float:
    """
    Relative entropy (another name for KL divergence).

    Args:
        p: First distribution
        q: Second distribution

    Returns:
        Relative entropy D_KL(p || q)
    """
    return kl_divergence(p, q, base=math.e)
