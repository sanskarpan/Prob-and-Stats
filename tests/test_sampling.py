"""
Unit tests for sampling module — Monte Carlo, MCMC, and related methods.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import math
import random

from sampling import (
    monte_carlo_sample,
    monte_carlo_integration,
    inverse_transform_sampling,
    rejection_sampling,
    importance_sampling,
    metropolis_hastings,
    gibbs_sampling,
    slice_sampling,
    hamiltonian_monte_carlo,
    langevin_dynamics,
    parallel_tempering,
    adaptive_metropolis,
)


def _stats(samples):
    """Compute mean and std of a sample list."""
    n = len(samples)
    m = sum(samples) / n
    v = sum((x - m) ** 2 for x in samples) / n
    return m, math.sqrt(v)


class TestMonteCarloSample(unittest.TestCase):

    def test_returns_correct_count(self):
        samples = monte_carlo_sample(lambda: random.gauss(0, 1), 500)
        self.assertEqual(len(samples), 500)

    def test_samples_are_numbers(self):
        samples = monte_carlo_sample(lambda: random.random(), 100)
        self.assertTrue(all(isinstance(s, float) for s in samples))


class TestMonteCarloIntegration(unittest.TestCase):

    def test_integral_x_squared_zero_to_one(self):
        # ∫₀¹ x² dx = 1/3
        random.seed(42)
        result = monte_carlo_integration(lambda x: x ** 2, 0, 1, n_samples=50000)
        self.assertAlmostEqual(result, 1.0 / 3.0, delta=0.01)

    def test_integral_constant(self):
        # ∫₀¹ 1 dx = 1
        random.seed(42)
        result = monte_carlo_integration(lambda x: 1.0, 0, 1, n_samples=1000)
        self.assertAlmostEqual(result, 1.0, delta=0.01)

    def test_integral_linear(self):
        # ∫₀² x dx = 2
        random.seed(42)
        result = monte_carlo_integration(lambda x: x, 0, 2, n_samples=50000)
        self.assertAlmostEqual(result, 2.0, delta=0.05)


class TestInverseTransformSampling(unittest.TestCase):

    def test_exponential_via_inverse_transform(self):
        # Exponential(λ=1): F^{-1}(u) = -log(1 - u)
        random.seed(42)
        inv_cdf = lambda u: -math.log(1 - u + 1e-300)
        samples = inverse_transform_sampling(inv_cdf, n_samples=2000)
        m, s = _stats(samples)
        self.assertAlmostEqual(m, 1.0, delta=0.1)  # mean = 1/λ = 1
        self.assertAlmostEqual(s, 1.0, delta=0.1)  # std = 1/λ = 1


class TestRejectionSampling(unittest.TestCase):

    def test_beta_via_rejection(self):
        # Sample from Beta(2, 2) using Uniform(0, 1)
        random.seed(42)
        target = lambda x: 6 * x * (1 - x)   # Unnormalized Beta(2,2) PDF
        proposal = lambda: random.random()
        proposal_pdf = lambda x: 1.0
        samples = rejection_sampling(target, proposal, proposal_pdf, M=1.6, n_samples=500)
        self.assertGreater(len(samples), 0)
        m, _ = _stats(samples)
        # Beta(2,2) mean = 0.5
        self.assertAlmostEqual(m, 0.5, delta=0.1)

    def test_warning_on_insufficient_samples(self):
        # M too large → low acceptance → fewer samples generated
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        target = lambda x: 0.01
        proposal = lambda: random.random()
        proposal_pdf = lambda x: 1.0
        with redirect_stdout(buf):
            samples = rejection_sampling(target, proposal, proposal_pdf,
                                          M=1.0, n_samples=1000, max_iterations=100)
        # Should have fewer samples than requested and print a warning
        self.assertLess(len(samples), 1000)
        self.assertIn("Warning", buf.getvalue())


class TestImportanceSampling(unittest.TestCase):

    def test_estimate_mean_under_normal(self):
        # Estimate E_{N(0,1)}[x²] = 1.0 using samples from N(0, 2)
        random.seed(42)
        f = lambda x: x ** 2
        target_pdf = lambda x: math.exp(-x**2 / 2) / math.sqrt(2 * math.pi)
        proposal_sample = lambda: random.gauss(0, 2)
        proposal_pdf = lambda x: math.exp(-x**2 / 8) / math.sqrt(8 * math.pi)
        est, ess = importance_sampling(f, target_pdf, proposal_sample, proposal_pdf, 2000)
        self.assertAlmostEqual(est, 1.0, delta=0.15)
        self.assertGreater(ess, 10)


class TestMetropolisHastings(unittest.TestCase):

    def test_normal_target(self):
        random.seed(42)
        target_log = lambda x: -0.5 * x ** 2
        proposal = lambda x: x + random.gauss(0, 0.5)
        samples = metropolis_hastings(target_log, proposal, 0.0,
                                      n_samples=3000, burn_in=1000)
        m, s = _stats(samples)
        self.assertAlmostEqual(m, 0.0, delta=0.15)
        self.assertAlmostEqual(s, 1.0, delta=0.15)

    def test_no_crash_with_extreme_random(self):
        # Previously crashed when random.random() returned 0.0
        import unittest.mock as mock
        target_log = lambda x: -0.5 * x ** 2
        proposal = lambda x: x + random.gauss(0, 0.5)
        # Inject 0.0 into the random stream; must not raise ValueError
        values = [0.0] + [0.5] * 500
        with mock.patch('random.random', side_effect=values):
            try:
                metropolis_hastings(target_log, proposal, 0.0, n_samples=5, burn_in=5)
            except StopIteration:
                pass  # side_effect exhausted — that's fine; no ValueError is the point

    def test_burn_in_discarded(self):
        # n_samples=10 should return exactly 10 samples
        random.seed(0)
        target_log = lambda x: -0.5 * x ** 2
        proposal = lambda x: x + random.gauss(0, 1)
        samples = metropolis_hastings(target_log, proposal, 0.0,
                                      n_samples=10, burn_in=100)
        self.assertEqual(len(samples), 10)


class TestGibbsSampling(unittest.TestCase):

    def test_bivariate_normal(self):
        # x|y ~ N(0.5y, 0.75), y|x ~ N(0.5x, 0.75)
        # Marginals: both N(0, 1)
        random.seed(42)
        samplers = [
            lambda state: random.gauss(0.5 * state[1], math.sqrt(0.75)),
            lambda state: random.gauss(0.5 * state[0], math.sqrt(0.75)),
        ]
        samples = gibbs_sampling(samplers, [0.0, 0.0], n_samples=2000, burn_in=500)
        self.assertEqual(len(samples), 2000)
        xs = [s[0] for s in samples]
        ys = [s[1] for s in samples]
        mx, sx = _stats(xs)
        my, sy = _stats(ys)
        self.assertAlmostEqual(mx, 0.0, delta=0.1)
        self.assertAlmostEqual(sx, 1.0, delta=0.15)
        self.assertAlmostEqual(my, 0.0, delta=0.1)
        self.assertAlmostEqual(sy, 1.0, delta=0.15)


class TestSliceSampling(unittest.TestCase):

    def test_normal_target(self):
        random.seed(42)
        log_pdf = lambda x: -0.5 * x ** 2
        samples = slice_sampling(log_pdf, 0.0, width=2.0, n_samples=2000, burn_in=300)
        m, s = _stats(samples)
        self.assertAlmostEqual(m, 0.0, delta=0.1)
        self.assertAlmostEqual(s, 1.0, delta=0.15)


class TestHamiltonianMonteCarlo(unittest.TestCase):

    def test_normal_target(self):
        random.seed(42)
        log_pdf = lambda x: -0.5 * x ** 2
        grad_log = lambda x: -x
        samples = hamiltonian_monte_carlo(log_pdf, grad_log, 0.0,
                                          epsilon=0.2, L=10, n_samples=1000, burn_in=200)
        m, s = _stats(samples)
        self.assertAlmostEqual(m, 0.0, delta=0.15)
        self.assertAlmostEqual(s, 1.0, delta=0.15)

    def test_leapfrog_reversibility(self):
        # HMC should produce unbiased samples; variance should be ~1 for N(0,1)
        random.seed(7)
        log_pdf = lambda x: -0.5 * x ** 2
        grad_log = lambda x: -x
        samples = hamiltonian_monte_carlo(log_pdf, grad_log, 1.0,
                                          epsilon=0.1, L=5, n_samples=500, burn_in=100)
        _, s = _stats(samples)
        self.assertAlmostEqual(s, 1.0, delta=0.2)


class TestLangevinDynamics(unittest.TestCase):

    def test_normal_target(self):
        random.seed(42)
        grad_log = lambda x: -x
        samples = langevin_dynamics(grad_log, 0.0, epsilon=0.05,
                                     n_samples=3000, burn_in=300)
        m, s = _stats(samples)
        self.assertAlmostEqual(m, 0.0, delta=0.15)
        self.assertAlmostEqual(s, 1.0, delta=0.15)


class TestParallelTempering(unittest.TestCase):

    def test_normal_target(self):
        random.seed(42)
        log_pdf = lambda x: -0.5 * x ** 2
        proposal = lambda x: x + random.gauss(0, 0.5)
        temps = [1.0, 2.0, 4.0]
        states = [0.0, 0.0, 0.0]
        samples = parallel_tempering(log_pdf, states, temps, proposal,
                                      n_samples=2000, burn_in=500)
        m, s = _stats(samples)
        self.assertAlmostEqual(m, 0.0, delta=0.15)
        self.assertAlmostEqual(s, 1.0, delta=0.2)

    def test_swap_formula_no_crash(self):
        # Ensure the swap logic (log(random+1e-300)) does not crash
        random.seed(0)
        log_pdf = lambda x: -0.5 * x ** 2
        proposal = lambda x: x + random.gauss(0, 1.0)
        samples = parallel_tempering(log_pdf, [0.0, 0.0], [1.0, 2.0], proposal,
                                      n_samples=50, burn_in=10)
        self.assertEqual(len(samples), 50)


class TestAdaptiveMetropolis(unittest.TestCase):

    def test_univariate_normal(self):
        random.seed(42)
        log_pdf = lambda x: -0.5 * x[0] ** 2
        samples = adaptive_metropolis(log_pdf, [0.0], n_samples=2000, burn_in=500)
        xs = [s[0] for s in samples]
        m, s = _stats(xs)
        self.assertAlmostEqual(m, 0.0, delta=0.15)
        self.assertAlmostEqual(s, 1.0, delta=0.2)

    def test_bivariate_normal_per_dim_scales(self):
        # Target: N([0,0], diag([1, 100])) — very different scales per dimension
        random.seed(42)
        log_pdf = lambda x: -0.5 * x[0] ** 2 - 0.5 * (x[1] / 10) ** 2
        samples = adaptive_metropolis(log_pdf, [0.0, 0.0], n_samples=3000, burn_in=1000,
                                       adaptation_interval=100)
        xs = [s[0] for s in samples]
        ys = [s[1] for s in samples]
        mx, sx = _stats(xs)
        my, sy = _stats(ys)
        self.assertAlmostEqual(mx, 0.0, delta=0.2)
        self.assertAlmostEqual(my, 0.0, delta=2.0)
        # With per-dim adaptation, y dimension (scale=10) should mix too
        self.assertGreater(sy, 3.0)  # std should be ~10, not stuck near 0


if __name__ == '__main__':
    unittest.main()
