"""
Probability Distributions
=========================

Implementation of common probability distributions from scratch.

Discrete distributions: Bernoulli, Binomial, Poisson, Categorical
Continuous distributions: Uniform, Normal, Exponential, Beta, Gamma

Applications in ML/DL:
- Normal distribution for weight initialization
- Bernoulli for binary classification
- Categorical for multi-class problems
- Exponential for survival analysis
- Beta/Gamma for Bayesian priors
"""

import math
import random
from typing import List, Optional, Union
from abc import ABC, abstractmethod


class Distribution(ABC):
    """Base class for probability distributions."""

    @abstractmethod
    def sample(self, n: int = 1) -> Union[float, List[float]]:
        """Generate random samples from the distribution."""
        pass

    @abstractmethod
    def mean(self) -> float:
        """Compute the expected value."""
        pass

    @abstractmethod
    def variance(self) -> float:
        """Compute the variance."""
        pass

    def std(self) -> float:
        """Compute the standard deviation."""
        return math.sqrt(self.variance())


class Bernoulli(Distribution):
    """
    Bernoulli Distribution - Binary outcomes (0 or 1)

    PMF: P(X=1) = p, P(X=0) = 1-p

    Used in: Binary classification, coin flips, binary events
    """

    def __init__(self, p: float):
        """
        Args:
            p: Probability of success (getting 1)
        """
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0, 1]")
        self.p = p

    def pmf(self, k: int) -> float:
        """Probability mass function."""
        if k == 1:
            return self.p
        elif k == 0:
            return 1 - self.p
        else:
            return 0.0

    def cdf(self, k: int) -> float:
        """Cumulative distribution function."""
        if k < 0:
            return 0.0
        elif k < 1:
            return 1 - self.p
        else:
            return 1.0

    def sample(self, n: int = 1) -> Union[int, List[int]]:
        """Generate random samples."""
        samples = [1 if random.random() < self.p else 0 for _ in range(n)]
        return samples[0] if n == 1 else samples

    def mean(self) -> float:
        """E[X] = p"""
        return self.p

    def variance(self) -> float:
        """Var(X) = p(1-p)"""
        return self.p * (1 - self.p)

    def __repr__(self) -> str:
        return f"Bernoulli(p={self.p})"


class Binomial(Distribution):
    """
    Binomial Distribution - Number of successes in n trials

    PMF: P(X=k) = C(n,k) * p^k * (1-p)^(n-k)

    Used in: A/B testing, quality control, repeated trials
    """

    def __init__(self, n: int, p: float):
        """
        Args:
            n: Number of trials
            p: Probability of success in each trial
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0, 1]")
        self.n = n
        self.p = p

    def _binomial_coefficient(self, n: int, k: int) -> int:
        """Compute C(n, k) = n! / (k! * (n-k)!)"""
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1

        # Use multiplicative formula for efficiency
        result = 1
        for i in range(min(k, n - k)):
            result = result * (n - i) // (i + 1)
        return result

    def pmf(self, k: int) -> float:
        """Probability mass function."""
        if k < 0 or k > self.n:
            return 0.0
        coef = self._binomial_coefficient(self.n, k)
        return coef * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k: int) -> float:
        """Cumulative distribution function."""
        if k < 0:
            return 0.0
        if k >= self.n:
            return 1.0
        return sum(self.pmf(i) for i in range(int(k) + 1))

    def sample(self, n: int = 1) -> Union[int, List[int]]:
        """Generate random samples."""
        samples = [sum(1 for _ in range(self.n) if random.random() < self.p)
                   for _ in range(n)]
        return samples[0] if n == 1 else samples

    def mean(self) -> float:
        """E[X] = np"""
        return self.n * self.p

    def variance(self) -> float:
        """Var(X) = np(1-p)"""
        return self.n * self.p * (1 - self.p)

    def __repr__(self) -> str:
        return f"Binomial(n={self.n}, p={self.p})"


class Poisson(Distribution):
    """
    Poisson Distribution - Number of events in fixed interval

    PMF: P(X=k) = (λ^k * e^(-λ)) / k!

    Used in: Count data, rare events, queueing theory
    """

    def __init__(self, lam: float):
        """
        Args:
            lam: Rate parameter (λ > 0)
        """
        if lam <= 0:
            raise ValueError("λ must be positive")
        self.lam = lam

    def pmf(self, k: int) -> float:
        """Probability mass function."""
        if k < 0:
            return 0.0
        return (self.lam ** k) * math.exp(-self.lam) / math.factorial(k)

    def cdf(self, k: int) -> float:
        """Cumulative distribution function."""
        if k < 0:
            return 0.0
        return sum(self.pmf(i) for i in range(int(k) + 1))

    def sample(self, n: int = 1) -> Union[int, List[int]]:
        """Generate random samples using Knuth's algorithm."""
        samples = []
        for _ in range(n):
            L = math.exp(-self.lam)
            k = 0
            p = 1.0
            while p > L:
                k += 1
                p *= random.random()
            samples.append(k - 1)
        return samples[0] if n == 1 else samples

    def mean(self) -> float:
        """E[X] = λ"""
        return self.lam

    def variance(self) -> float:
        """Var(X) = λ"""
        return self.lam

    def __repr__(self) -> str:
        return f"Poisson(lam={self.lam})"


class Categorical(Distribution):
    """
    Categorical Distribution - Discrete outcomes with K categories

    PMF: P(X=k) = p_k

    Used in: Multi-class classification, discrete choice models
    """

    def __init__(self, probs: List[float]):
        """
        Args:
            probs: List of probabilities for each category (must sum to 1)
        """
        if not math.isclose(sum(probs), 1.0, rel_tol=1e-9):
            raise ValueError("Probabilities must sum to 1")
        if any(p < 0 for p in probs):
            raise ValueError("All probabilities must be non-negative")
        self.probs = probs
        self.k = len(probs)

    def pmf(self, category: int) -> float:
        """Probability mass function."""
        if 0 <= category < self.k:
            return self.probs[category]
        return 0.0

    def sample(self, n: int = 1) -> Union[int, List[int]]:
        """Generate random samples."""
        samples = []
        for _ in range(n):
            r = random.random()
            cumsum = 0.0
            for i, p in enumerate(self.probs):
                cumsum += p
                if r <= cumsum:
                    samples.append(i)
                    break
        return samples[0] if n == 1 else samples

    def mean(self) -> float:
        """E[X] = Σ k * p_k"""
        return sum(k * p for k, p in enumerate(self.probs))

    def variance(self) -> float:
        """Var(X) = Σ k² * p_k - (E[X])²"""
        mean_val = self.mean()
        return sum(k**2 * p for k, p in enumerate(self.probs)) - mean_val**2

    def entropy(self) -> float:
        """Shannon entropy H(X) = -Σ p_k * log(p_k)"""
        return -sum(p * math.log(p) if p > 0 else 0 for p in self.probs)

    def __repr__(self) -> str:
        return f"Categorical(probs={self.probs})"


class Uniform(Distribution):
    """
    Continuous Uniform Distribution - Equal probability over interval

    PDF: f(x) = 1/(b-a) for x in [a, b]

    Used in: Random initialization, sampling
    """

    def __init__(self, a: float = 0.0, b: float = 1.0):
        """
        Args:
            a: Lower bound
            b: Upper bound (b > a)
        """
        if b <= a:
            raise ValueError("b must be greater than a")
        self.a = a
        self.b = b

    def pdf(self, x: float) -> float:
        """Probability density function."""
        if self.a <= x <= self.b:
            return 1.0 / (self.b - self.a)
        return 0.0

    def cdf(self, x: float) -> float:
        """Cumulative distribution function."""
        if x < self.a:
            return 0.0
        elif x > self.b:
            return 1.0
        else:
            return (x - self.a) / (self.b - self.a)

    def sample(self, n: int = 1) -> Union[float, List[float]]:
        """Generate random samples."""
        samples = [self.a + random.random() * (self.b - self.a) for _ in range(n)]
        return samples[0] if n == 1 else samples

    def mean(self) -> float:
        """E[X] = (a + b) / 2"""
        return (self.a + self.b) / 2

    def variance(self) -> float:
        """Var(X) = (b - a)² / 12"""
        return ((self.b - self.a) ** 2) / 12

    def __repr__(self) -> str:
        return f"Uniform(a={self.a}, b={self.b})"


class Normal(Distribution):
    """
    Normal (Gaussian) Distribution

    PDF: f(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))

    Most important distribution in ML/DL:
    - Central limit theorem
    - Weight initialization
    - Noise modeling
    - Gaussian processes
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        """
        Args:
            mu: Mean
            sigma: Standard deviation (σ > 0)
        """
        if sigma <= 0:
            raise ValueError("σ must be positive")
        self.mu = mu
        self.sigma = sigma

    def pdf(self, x: float) -> float:
        """Probability density function."""
        coefficient = 1.0 / (self.sigma * math.sqrt(2 * math.pi))
        exponent = -((x - self.mu) ** 2) / (2 * self.sigma ** 2)
        return coefficient * math.exp(exponent)

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function using error function approximation.
        """
        # Standardize
        z = (x - self.mu) / self.sigma
        # Use error function approximation
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    def sample(self, n: int = 1) -> Union[float, List[float]]:
        """Generate random samples using Box-Muller transform."""
        samples = []
        for i in range(0, n, 2):
            # Box-Muller transform
            u1 = random.random()
            u2 = random.random()

            z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)

            samples.append(self.mu + z0 * self.sigma)
            if len(samples) < n:
                samples.append(self.mu + z1 * self.sigma)

        return samples[0] if n == 1 else samples[:n]

    def mean(self) -> float:
        """E[X] = μ"""
        return self.mu

    def variance(self) -> float:
        """Var(X) = σ²"""
        return self.sigma ** 2

    def __repr__(self) -> str:
        return f"Normal(mu={self.mu}, sigma={self.sigma})"

    def quantile(self, p: float) -> float:
        """
        Compute quantile (inverse CDF) via the inverse error function.

        Args:
            p: Probability (0 < p < 1)

        Returns:
            x such that P(X ≤ x) = p
        """
        if not 0 < p < 1:
            raise ValueError("p must be in (0, 1)")
        # Φ^{-1}(p) = √2 · erfinv(2p - 1)
        return self.mu + self.sigma * math.sqrt(2) * math.erfinv(2 * p - 1)


class Exponential(Distribution):
    """
    Exponential Distribution - Time until event occurs

    PDF: f(x) = λ * e^(-λx) for x ≥ 0

    Used in: Survival analysis, queueing theory, time-to-event
    """

    def __init__(self, lam: float = 1.0):
        """
        Args:
            lam: Rate parameter (λ > 0)
        """
        if lam <= 0:
            raise ValueError("λ must be positive")
        self.lam = lam

    def pdf(self, x: float) -> float:
        """Probability density function."""
        if x < 0:
            return 0.0
        return self.lam * math.exp(-self.lam * x)

    def cdf(self, x: float) -> float:
        """Cumulative distribution function."""
        if x < 0:
            return 0.0
        return 1 - math.exp(-self.lam * x)

    def sample(self, n: int = 1) -> Union[float, List[float]]:
        """Generate random samples using inverse transform."""
        samples = [-math.log(1 - random.random()) / self.lam for _ in range(n)]
        return samples[0] if n == 1 else samples

    def mean(self) -> float:
        """E[X] = 1/λ"""
        return 1.0 / self.lam

    def variance(self) -> float:
        """Var(X) = 1/λ²"""
        return 1.0 / (self.lam ** 2)

    def __repr__(self) -> str:
        return f"Exponential(lam={self.lam})"


class Beta(Distribution):
    """
    Beta Distribution - Continuous distribution on [0, 1]

    PDF: f(x) = x^(α-1) * (1-x)^(β-1) / B(α, β)

    Used in: Bayesian inference (priors for probabilities), A/B testing
    """

    def __init__(self, alpha: float, beta: float):
        """
        Args:
            alpha: Shape parameter (α > 0)
            beta: Shape parameter (β > 0)
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("α and β must be positive")
        self.alpha = alpha
        self.beta = beta

    def _beta_function(self, a: float, b: float) -> float:
        """Compute Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b)"""
        return math.gamma(a) * math.gamma(b) / math.gamma(a + b)

    def pdf(self, x: float) -> float:
        """Probability density function."""
        if not 0 <= x <= 1:
            return 0.0
        if x == 0:
            return float('inf') if self.alpha < 1 else (0.0 if self.alpha > 1 else 1.0)
        if x == 1:
            return float('inf') if self.beta < 1 else (0.0 if self.beta > 1 else 1.0)

        numerator = (x ** (self.alpha - 1)) * ((1 - x) ** (self.beta - 1))
        denominator = self._beta_function(self.alpha, self.beta)
        return numerator / denominator

    def sample(self, n: int = 1) -> Union[float, List[float]]:
        """
        Generate random samples using ratio of Gamma distributions.
        Beta(α, β) = Gamma(α) / (Gamma(α) + Gamma(β))
        """
        gamma_alpha = Gamma(self.alpha, 1.0)
        gamma_beta = Gamma(self.beta, 1.0)

        samples = []
        for _ in range(n):
            x = gamma_alpha.sample()
            y = gamma_beta.sample()
            samples.append(x / (x + y))

        return samples[0] if n == 1 else samples

    def mean(self) -> float:
        """E[X] = α / (α + β)"""
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        """Var(X) = αβ / ((α+β)²(α+β+1))"""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def __repr__(self) -> str:
        return f"Beta(alpha={self.alpha}, beta={self.beta})"


class Gamma(Distribution):
    """
    Gamma Distribution - Generalization of exponential

    PDF: f(x) = (β^α / Γ(α)) * x^(α-1) * e^(-βx)

    Used in: Bayesian inference, waiting times, prior distributions
    """

    def __init__(self, alpha: float, beta: float = 1.0):
        """
        Args:
            alpha: Shape parameter (α > 0)
            beta: Rate parameter (β > 0)
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("α and β must be positive")
        self.alpha = alpha
        self.beta = beta

    def pdf(self, x: float) -> float:
        """Probability density function."""
        if x < 0:
            return 0.0
        if x == 0:
            return float('inf') if self.alpha < 1 else (0.0 if self.alpha > 1 else self.beta)

        coefficient = (self.beta ** self.alpha) / math.gamma(self.alpha)
        return coefficient * (x ** (self.alpha - 1)) * math.exp(-self.beta * x)

    def sample(self, n: int = 1) -> Union[float, List[float]]:
        """
        Generate random samples using Marsaglia and Tsang's method.
        """
        samples = []

        for _ in range(n):
            # For α < 1, use rejection method
            if self.alpha < 1:
                # Use Gamma(α+1) then scale
                alpha_temp = self.alpha + 1
                d = alpha_temp - 1/3
                c = 1 / math.sqrt(9 * d)

                while True:
                    x = random.gauss(0, 1)
                    v = (1 + c * x) ** 3
                    if v > 0:
                        u = random.random()
                        if u < 1 - 0.0331 * x**4:
                            sample = d * v / self.beta
                            # Scale for original alpha
                            sample *= random.random() ** (1/self.alpha)
                            samples.append(sample)
                            break
                        if math.log(u) < 0.5 * x**2 + d * (1 - v + math.log(v)):
                            sample = d * v / self.beta
                            sample *= random.random() ** (1/self.alpha)
                            samples.append(sample)
                            break
            else:
                # Marsaglia and Tsang's method for α ≥ 1
                d = self.alpha - 1/3
                c = 1 / math.sqrt(9 * d)

                while True:
                    x = random.gauss(0, 1)
                    v = (1 + c * x) ** 3
                    if v > 0:
                        u = random.random()
                        if u < 1 - 0.0331 * x**4:
                            samples.append(d * v / self.beta)
                            break
                        if math.log(u) < 0.5 * x**2 + d * (1 - v + math.log(v)):
                            samples.append(d * v / self.beta)
                            break

        return samples[0] if n == 1 else samples

    def mean(self) -> float:
        """E[X] = α/β"""
        return self.alpha / self.beta

    def variance(self) -> float:
        """Var(X) = α/β²"""
        return self.alpha / (self.beta ** 2)

    def __repr__(self) -> str:
        return f"Gamma(alpha={self.alpha}, beta={self.beta})"
