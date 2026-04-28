"""
Statistics
==========

Descriptive statistics, hypothesis testing, and statistical inference.

Applications in ML/DL:
- Data preprocessing and analysis
- Feature selection
- Model evaluation
- A/B testing
- Significance testing
"""

import math
from typing import List, Tuple, Optional, Union
from collections import Counter


# ── Private mathematical helpers ───────────────────────────────────────────────

def _betainc(x: float, a: float, b: float) -> float:
    """
    Regularized incomplete beta function I_x(a, b).
    Modified Lentz's continued fraction — NR 6.4 (betacf).
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # Use symmetry relation to keep x in the converging half of the CF
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(1.0 - x, b, a)
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - log_beta) / a
    # Modified Lentz (NR betacf): c starts at 1.0, d starts at 1/b_1
    TINY = 1e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < TINY:
        d = TINY
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        # Even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < TINY:
            d = TINY
        c = 1.0 + aa / c
        if abs(c) < TINY:
            c = TINY
        d = 1.0 / d
        h *= d * c
        # Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < TINY:
            d = TINY
        c = 1.0 + aa / c
        if abs(c) < TINY:
            c = TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-9:
            break
    return front * h


def _t_ppf(p: float, df: int) -> float:
    """
    Quantile (inverse CDF) of the t-distribution at probability p via binary search.
    CDF(t; df) = 1 - I_{df/(df+t²)}(df/2, 1/2) / 2
    """
    lo, hi = 0.0, 1000.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        x = df / (df + mid * mid)
        cdf_val = 1.0 - _betainc(x, df / 2.0, 0.5) / 2.0
        if cdf_val < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _gammainc(a: float, x: float) -> float:
    """
    Regularized lower incomplete gamma P(a, x) = γ(a,x)/Γ(a).
    Series expansion for x < a+1, Lentz continued fraction otherwise.
    """
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        for n in range(1, 500):
            term *= x / (a + n)
            total += term
            if term < total * 1e-12:
                break
        return math.exp(-x + a * math.log(x) - math.lgamma(a)) * total
    else:
        TINY = 1e-30
        f = 1.0 / (x + 1.0 - a)
        C = 1.0 / TINY
        D = f
        for i in range(1, 300):
            an = -i * (i - a)
            bn = x + 2.0 * i + 1.0 - a
            D = bn + an * D
            if abs(D) < TINY:
                D = TINY
            C = bn + an / C
            if abs(C) < TINY:
                C = TINY
            D = 1.0 / D
            delta = C * D
            f *= delta
            if abs(delta - 1.0) < 1e-10:
                break
        q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * f
        return 1.0 - q


def _chi2_ppf(p: float, df: int) -> float:
    """
    Quantile of chi-squared(df) distribution.
    chi2(df) CDF(x) = P(df/2, x/2) where P is the regularized lower incomplete gamma.
    """
    lo, hi = 0.0, max(100.0, float(df) * 5.0)
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _gammainc(df / 2.0, mid / 2.0) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def mean(data: List[float]) -> float:
    """
    Compute arithmetic mean (average).

    Args:
        data: List of numerical values

    Returns:
        Mean value

    Example:
        >>> mean([1, 2, 3, 4, 5])
        3.0
    """
    if not data:
        raise ValueError("Data cannot be empty")
    return sum(data) / len(data)


def median(data: List[float]) -> float:
    """
    Compute median (50th percentile).

    Args:
        data: List of numerical values

    Returns:
        Median value

    Example:
        >>> median([1, 2, 3, 4, 5])
        3.0
        >>> median([1, 2, 3, 4])
        2.5
    """
    if not data:
        raise ValueError("Data cannot be empty")

    sorted_data = sorted(data)
    n = len(sorted_data)

    if n % 2 == 1:
        return sorted_data[n // 2]
    else:
        return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2


def mode(data: List[Union[float, int, str]]) -> Union[float, int, str, List]:
    """
    Compute mode (most frequent value).

    Args:
        data: List of values

    Returns:
        Most frequent value(s)

    Example:
        >>> mode([1, 2, 2, 3, 4])
        2
    """
    if not data:
        raise ValueError("Data cannot be empty")

    counter = Counter(data)
    max_count = max(counter.values())
    modes = [value for value, count in counter.items() if count == max_count]

    return modes[0] if len(modes) == 1 else modes


def quantile(data: List[float], q: float) -> float:
    """
    Compute quantile (generalized percentile).

    Args:
        data: List of numerical values
        q: Quantile to compute (0 ≤ q ≤ 1)

    Returns:
        Quantile value

    Example:
        >>> quantile([1, 2, 3, 4, 5], 0.5)  # Median
        3.0
        >>> quantile([1, 2, 3, 4, 5], 0.25)  # 25th percentile
        2.0
    """
    if not 0 <= q <= 1:
        raise ValueError("Quantile must be in [0, 1]")
    if not data:
        raise ValueError("Data cannot be empty")

    sorted_data = sorted(data)
    n = len(sorted_data)
    index = q * (n - 1)

    if index.is_integer():
        return sorted_data[int(index)]
    else:
        lower = sorted_data[int(math.floor(index))]
        upper = sorted_data[int(math.ceil(index))]
        fraction = index - math.floor(index)
        return lower + fraction * (upper - lower)


def variance(data: List[float], ddof: int = 0) -> float:
    """
    Compute variance.

    Args:
        data: List of numerical values
        ddof: Delta degrees of freedom (0 for population, 1 for sample)

    Returns:
        Variance

    Example:
        >>> variance([1, 2, 3, 4, 5], ddof=0)  # Population variance
        2.0
        >>> variance([1, 2, 3, 4, 5], ddof=1)  # Sample variance
        2.5
    """
    if not data:
        raise ValueError("Data cannot be empty")
    if len(data) <= ddof:
        raise ValueError("Insufficient data for given ddof")

    mean_val = mean(data)
    squared_diff = [(x - mean_val) ** 2 for x in data]
    return sum(squared_diff) / (len(data) - ddof)


def std(data: List[float], ddof: int = 0) -> float:
    """
    Compute standard deviation.

    Args:
        data: List of numerical values
        ddof: Delta degrees of freedom

    Returns:
        Standard deviation
    """
    return math.sqrt(variance(data, ddof))


def covariance(x: List[float], y: List[float], ddof: int = 0) -> float:
    """
    Compute covariance between two variables.

    Args:
        x: First variable
        y: Second variable
        ddof: Delta degrees of freedom

    Returns:
        Covariance

    Example:
        >>> x = [1, 2, 3, 4, 5]
        >>> y = [2, 4, 6, 8, 10]
        >>> covariance(x, y, ddof=0)
        4.0
    """
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    if not x:
        raise ValueError("Data cannot be empty")
    if len(x) <= ddof:
        raise ValueError("Insufficient data for given ddof")

    mean_x = mean(x)
    mean_y = mean(y)

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    return cov / (len(x) - ddof)


def correlation(x: List[float], y: List[float]) -> float:
    """
    Compute Pearson correlation coefficient.

    Args:
        x: First variable
        y: Second variable

    Returns:
        Correlation coefficient (-1 to 1)

    Example:
        >>> x = [1, 2, 3, 4, 5]
        >>> y = [2, 4, 6, 8, 10]
        >>> correlation(x, y)
        1.0
    """
    if len(x) != len(y):
        raise ValueError("x and y must have same length")

    cov = covariance(x, y, ddof=1)
    std_x = std(x, ddof=1)
    std_y = std(y, ddof=1)

    if std_x == 0 or std_y == 0:
        raise ValueError("Cannot compute correlation with zero variance")

    return cov / (std_x * std_y)


def z_score(data: List[float], value: float) -> float:
    """
    Compute z-score (standardized score).

    z = (x - μ) / σ

    Args:
        data: Dataset
        value: Value to standardize

    Returns:
        Z-score

    Example:
        >>> z_score([1, 2, 3, 4, 5], 5)
        1.414...
    """
    mean_val = mean(data)
    std_val = std(data, ddof=1)

    if std_val == 0:
        raise ValueError("Cannot compute z-score with zero variance")

    return (value - mean_val) / std_val


def standard_error(data: List[float]) -> float:
    """
    Compute standard error of the mean.

    SE = σ / √n

    Args:
        data: Dataset

    Returns:
        Standard error
    """
    if not data:
        raise ValueError("Data cannot be empty")

    return std(data, ddof=1) / math.sqrt(len(data))


def confidence_interval(
    data: List[float],
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Compute confidence interval for the mean using t-distribution.

    Args:
        data: Dataset
        confidence: Confidence level (default 0.95 for 95%)

    Returns:
        Tuple of (lower bound, upper bound)

    Example:
        >>> ci = confidence_interval([1, 2, 3, 4, 5], 0.95)
    """
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be in (0, 1)")

    n = len(data)
    mean_val = mean(data)
    se = standard_error(data)

    # Use t-distribution critical value (exact for all n)
    df = n - 1
    t_crit = _t_ppf(1.0 - (1.0 - confidence) / 2.0, df)
    margin = t_crit * se
    return (mean_val - margin, mean_val + margin)


def t_statistic(sample: List[float], population_mean: float) -> float:
    """
    Compute t-statistic for one-sample t-test.

    t = (x̄ - μ) / (s / √n)

    Args:
        sample: Sample data
        population_mean: Hypothesized population mean

    Returns:
        T-statistic

    Example:
        >>> t_statistic([1, 2, 3, 4, 5], 2.5)
        1.118...
    """
    sample_mean = mean(sample)
    sample_std = std(sample, ddof=1)
    n = len(sample)

    return (sample_mean - population_mean) / (sample_std / math.sqrt(n))


def t_test(
    sample: List[float],
    population_mean: float,
    alpha: float = 0.05
) -> Tuple[float, bool]:
    """
    Perform one-sample t-test.

    Tests H₀: μ = population_mean vs H₁: μ ≠ population_mean

    Args:
        sample: Sample data
        population_mean: Hypothesized population mean
        alpha: Significance level

    Returns:
        Tuple of (t-statistic, reject_null)

    Example:
        >>> t, reject = t_test([1, 2, 3, 4, 5], 10)
        >>> reject
        True
    """
    t_stat = t_statistic(sample, population_mean)
    df = len(sample) - 1
    critical = _t_ppf(1.0 - alpha / 2.0, df)
    reject_null = abs(t_stat) > critical

    return t_stat, reject_null


def two_sample_t_test(
    sample1: List[float],
    sample2: List[float],
    alpha: float = 0.05
) -> Tuple[float, bool]:
    """
    Perform two-sample t-test (independent samples).

    Tests H₀: μ₁ = μ₂ vs H₁: μ₁ ≠ μ₂

    Args:
        sample1: First sample
        sample2: Second sample
        alpha: Significance level

    Returns:
        Tuple of (t-statistic, reject_null)

    Example:
        >>> s1 = [1, 2, 3, 4, 5]
        >>> s2 = [6, 7, 8, 9, 10]
        >>> t, reject = two_sample_t_test(s1, s2)
        >>> reject
        True
    """
    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = mean(sample1), mean(sample2)
    var1, var2 = variance(sample1, ddof=1), variance(sample2, ddof=1)

    # Pooled variance
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)

    # Standard error
    se = math.sqrt(pooled_var * (1/n1 + 1/n2))

    # T-statistic
    t_stat = (mean1 - mean2) / se

    # Exact t-distribution critical value
    df = n1 + n2 - 2
    critical = _t_ppf(1.0 - alpha / 2.0, df)
    reject_null = abs(t_stat) > critical

    return t_stat, reject_null


def chi_square_test(
    observed: List[int],
    expected: List[float],
    alpha: float = 0.05
) -> Tuple[float, bool]:
    """
    Perform chi-square goodness-of-fit test.

    χ² = Σ(Oᵢ - Eᵢ)² / Eᵢ

    Args:
        observed: Observed frequencies
        expected: Expected frequencies
        alpha: Significance level

    Returns:
        Tuple of (chi-square statistic, reject_null)

    Example:
        >>> obs = [10, 20, 30]
        >>> exp = [15, 20, 25]
        >>> chi2, reject = chi_square_test(obs, exp)
    """
    if len(observed) != len(expected):
        raise ValueError("Observed and expected must have same length")

    chi_square = sum((o - e) ** 2 / e for o, e in zip(observed, expected))

    # df = k - 1 where k is number of categories
    df = len(observed) - 1
    if df < 1:
        raise ValueError("Need at least 2 categories for chi-square test")

    # Exact critical value from chi-square distribution
    critical = _chi2_ppf(1.0 - alpha, df)
    reject_null = chi_square > critical

    return chi_square, reject_null


def z_test(
    sample: List[float],
    population_mean: float,
    population_std: float,
    alpha: float = 0.05
) -> Tuple[float, bool]:
    """
    Perform one-sample z-test.

    Used when population standard deviation is known.

    z = (x̄ - μ) / (σ / √n)

    Args:
        sample: Sample data
        population_mean: Hypothesized population mean
        population_std: Known population standard deviation
        alpha: Significance level

    Returns:
        Tuple of (z-statistic, reject_null)

    Example:
        >>> z, reject = z_test([1, 2, 3, 4, 5], 10, 2)
        >>> reject
        True
    """
    sample_mean = mean(sample)
    n = len(sample)

    z_stat = (sample_mean - population_mean) / (population_std / math.sqrt(n))

    # Exact two-tailed normal critical value: Φ^{-1}(1 - alpha/2) = √2 * erfinv(1 - alpha)
    critical = math.sqrt(2) * math.erfinv(1.0 - alpha)
    reject_null = abs(z_stat) > critical

    return z_stat, reject_null


def p_value_from_z(z: float) -> float:
    """
    Compute two-tailed p-value from z-statistic.

    Uses approximation for standard normal CDF.

    Args:
        z: Z-statistic

    Returns:
        P-value

    Example:
        >>> p_value_from_z(1.96)
        0.05  # approximately
    """
    # Using error function approximation
    # This is a simplified version
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return p


def effect_size_cohens_d(
    sample1: List[float],
    sample2: List[float]
) -> float:
    """
    Compute Cohen's d effect size.

    d = (μ₁ - μ₂) / σ_pooled

    Args:
        sample1: First sample
        sample2: Second sample

    Returns:
        Cohen's d (effect size)

    Interpretation:
        - Small: d ≈ 0.2
        - Medium: d ≈ 0.5
        - Large: d ≈ 0.8

    Example:
        >>> s1 = [1, 2, 3, 4, 5]
        >>> s2 = [6, 7, 8, 9, 10]
        >>> effect_size_cohens_d(s1, s2)
        3.162...
    """
    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = mean(sample1), mean(sample2)
    var1, var2 = variance(sample1, ddof=1), variance(sample2, ddof=1)

    # Pooled standard deviation
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    return (mean1 - mean2) / pooled_std


def ab_test(
    control: List[int],
    treatment: List[int],
    alpha: float = 0.05
) -> dict:
    """
    Perform A/B test for binary outcomes (conversion rates).

    Args:
        control: Control group outcomes (0s and 1s)
        treatment: Treatment group outcomes (0s and 1s)
        alpha: Significance level

    Returns:
        Dictionary with test results

    Example:
        >>> control = [0, 0, 1, 0, 1, 1, 0, 0, 1, 0]
        >>> treatment = [1, 1, 1, 0, 1, 1, 1, 0, 1, 1]
        >>> result = ab_test(control, treatment)
    """
    n_control = len(control)
    n_treatment = len(treatment)

    # Conversion rates
    p_control = sum(control) / n_control
    p_treatment = sum(treatment) / n_treatment

    # Pooled proportion
    p_pooled = (sum(control) + sum(treatment)) / (n_control + n_treatment)

    # Standard error
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))

    # Z-statistic
    if se == 0:
        z_stat = 0.0
    else:
        z_stat = (p_treatment - p_control) / se

    # Critical value
    critical = 1.96 if alpha == 0.05 else 2.576

    # Results
    return {
        'control_rate': p_control,
        'treatment_rate': p_treatment,
        'lift': (p_treatment - p_control) / p_control if p_control > 0 else 0,
        'z_statistic': z_stat,
        'p_value': p_value_from_z(z_stat),
        'significant': abs(z_stat) > critical,
        'reject_null': abs(z_stat) > critical
    }


def bootstrap_confidence_interval(
    data: List[float],
    statistic_func=mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for any statistic.

    Args:
        data: Original dataset
        statistic_func: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        Tuple of (lower bound, upper bound)

    Example:
        >>> import random
        >>> random.seed(42)
        >>> ci = bootstrap_confidence_interval([1, 2, 3, 4, 5])
    """
    import random

    bootstrap_stats = []
    n = len(data)

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = [data[random.randint(0, n-1)] for _ in range(n)]
        bootstrap_stats.append(statistic_func(sample))

    # Compute percentiles
    alpha = 1 - confidence
    lower_percentile = alpha / 2
    upper_percentile = 1 - alpha / 2

    lower_bound = quantile(bootstrap_stats, lower_percentile)
    upper_bound = quantile(bootstrap_stats, upper_percentile)

    return (lower_bound, upper_bound)
