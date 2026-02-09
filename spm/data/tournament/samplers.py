"""Tournament GCD algorithm and samplers for n = 4, 8, 16 inputs.

This module implements the core tournament GCD algorithm and sampler classes that generate
training data. The tournament GCD computes gcd(x₁, ..., xₙ) using a binary tree structure
where pairs are combined via extended Euclidean algorithm, propagating Bézout coefficients upward.

Core algorithm:
    tournament_gcd(inputs) → (k, coeffs, rounds)

    Example for [12, 18, 8, 24]:
        Round 0: egcd(12,18)=(6,-1,1), egcd(8,24)=(8,1,0)
                → coeffs [[-1,1,0,0], [0,0,1,0]]
        Round 1: egcd(6,8)=(2,-1,1)
                → coeffs = -1*[-1,1,0,0] + 1*[0,0,1,0] = [1,-1,1,0]
        Result: k=2, coeffs=[1,-1,1,0]
        Verify: 1×12 + (-1)×18 + 1×8 + 0×24 = 2 ✓

Sampler hierarchy:
    LogUniformTournamentGCDSampler(n, ubound) → TournamentSamples
        ↓ wraps
    TournamentTranscriptSampler → TournamentTranscript (adds coeffs)
        ↓ wraps
    TournamentAnnotatedTranscriptSampler → TournamentAnnotatedTranscript (adds round annotations)

The samplers generate batches of samples by:
1. Sampling n random integers from log-uniform distribution [1, ubound]
2. Running tournament_gcd() on each sample to compute k and coefficients
3. Optionally storing intermediate (g, u, v) values for each round (Option A annotations)

Set np.random.seed(67) before calling samplers for reproducibility. sixseven
"""

import abc
from typing import Tuple, List

import numpy as np

from spm.data.samples import DTYPE, fcast
from spm.utils import egcd


def tournament_gcd(inputs: List[int]) -> Tuple[int, np.ndarray, List[List[Tuple[int, int, int]]]]:
    """
    Compute GCD of n inputs using tournament structure.

    The tournament uses a binary tree where pairs are combined via extended
    Euclidean algorithm, propagating Bézout coefficients upward.

    Args:
        inputs: List of n integers (n must be power of 2: 4, 8, or 16)

    Returns:
        gcd: The GCD of all inputs
        coeffs: numpy array of n Bézout coefficients where sum(c_i * x_i) = gcd
        rounds: List of rounds, each containing (g, u, v) tuples for each pair.
                Used for Option A annotations.

    Example:
        >>> gcd, coeffs, rounds = tournament_gcd([12, 18, 8, 24])
        >>> gcd
        2
        >>> coeffs
        array([ 1, -1,  1,  0])
        >>> # Verify: 1*12 + (-1)*18 + 1*8 + 0*24 = 2
    """
    n = len(inputs)
    assert n in [4, 8, 16], f"n must be 4, 8, or 16, got {n}"
    assert n & (n - 1) == 0, "n must be a power of 2"

    # Initialise leaf nodes: each has (value, one-hot coefficient vector)
    layer = []
    for i, x in enumerate(inputs):
        one_hot = np.zeros(n, dtype=DTYPE)
        one_hot[i] = 1
        layer.append((x, one_hot))

    rounds = []  # Store intermediate results for annotations

    # Tournament rounds
    while len(layer) > 1:
        next_layer = []
        round_results = []

        for j in range(0, len(layer), 2):
            left_val, left_coeffs = layer[j]
            right_val, right_coeffs = layer[j + 1]

            g, u, v = egcd(left_val, right_val)

            # Store intermediate result for annotations
            round_results.append((g, u, v))

            # Combine coefficients: new_coeffs = u * left.coeffs + v * right.coeffs
            new_coeffs = u * left_coeffs + v * right_coeffs
            next_layer.append((g, new_coeffs))

        rounds.append(round_results)
        layer = next_layer

    final_gcd, final_coeffs = layer[0]
    return final_gcd, final_coeffs, rounds


# ---------------------------------------------------------------------------
# Sampler classes
# ---------------------------------------------------------------------------


class TournamentSampler(abc.ABC):
    """Abstract base class for tournament GCD samplers."""

    @abc.abstractmethod
    def sample(self, num_samples: int):
        raise NotImplementedError

    def __call__(self, num_samples: int):
        return self.sample(num_samples)


class TournamentGCDSampler(TournamentSampler):
    """Abstract class for samplers that sample n inputs and compute their GCD."""

    def __init__(self, n: int, ubound: int):
        assert n in [4, 8, 16], f"n must be 4, 8, or 16, got {n}"
        self.n = n
        self.ubound = ubound

    @abc.abstractmethod
    def sample_inputs(self, num_samples: int) -> np.ndarray:
        """Sample inputs of shape (num_samples, n)."""
        raise NotImplementedError

    def sample(self, num_samples: int):
        """Sample inputs and compute their GCDs."""
        from spm.data.tournament.samples import TournamentSamples

        if num_samples == 0:
            return TournamentSamples(
                self.n,
                np.array([], dtype=DTYPE).reshape(0, self.n),
                np.array([], dtype=DTYPE),
            )

        inputs = self.sample_inputs(num_samples)
        # Compute GCD using numpy's reduce
        k = np.gcd.reduce(inputs, axis=1)
        return TournamentSamples(self.n, inputs, k)


class LogUniformTournamentGCDSampler(TournamentGCDSampler):
    """Sample n integers from 1 to ubound with log-uniform distribution."""

    def __init__(self, n: int, ubound: int, base: int = 10):
        super().__init__(n, ubound)
        self.log_base = base

    def sample_inputs(self, num_samples: int) -> np.ndarray:
        log_ubound = np.emath.logn(self.log_base, self.ubound)
        log_inputs = np.random.uniform(0, log_ubound, size=(num_samples, self.n))
        inputs = fcast(np.power(self.log_base, log_inputs))
        # Ensure all values are at least 1
        inputs = np.maximum(inputs, 1)
        return inputs


class UniformTournamentGCDSampler(TournamentGCDSampler):
    """Sample n integers from 1 to ubound uniformly."""

    def sample_inputs(self, num_samples: int) -> np.ndarray:
        return np.random.randint(1, self.ubound + 1, size=(num_samples, self.n), dtype=DTYPE)


class TournamentTranscriptSampler(TournamentSampler):
    """Wraps a TournamentGCDSampler and computes Bézout coefficients."""

    def __init__(self, inner_sampler: TournamentGCDSampler):
        self.inner_sampler = inner_sampler
        self.n = inner_sampler.n

    def sample(self, num_samples: int):
        from spm.data.tournament.samples import TournamentTranscript

        samples = self.inner_sampler(num_samples)

        # Compute coefficients for each sample
        coeffs_list = []
        for i in range(len(samples)):
            inputs = samples.inputs[i].tolist()
            k, coeffs, _ = tournament_gcd(inputs)
            assert k == samples.k[i], f"GCD mismatch: {k} != {samples.k[i]}"
            coeffs_list.append(coeffs)

        coeffs = np.stack(coeffs_list, axis=0)
        return TournamentTranscript(self.n, samples.inputs, samples.k, coeffs)


class TournamentAnnotatedTranscriptSampler(TournamentSampler):
    """Wraps a TournamentGCDSampler and computes full annotations (Option A)."""

    def __init__(self, inner_sampler: TournamentGCDSampler):
        self.inner_sampler = inner_sampler
        self.n = inner_sampler.n
        self.num_rounds = int(np.log2(self.n))

    def sample(self, num_samples: int):
        from spm.data.tournament.samples import TournamentAnnotatedTranscript

        samples = self.inner_sampler(num_samples)

        # Initialise storage for annotations
        coeffs_list = []
        round_gcds = [[] for _ in range(self.num_rounds)]
        round_u = [[] for _ in range(self.num_rounds)]
        round_v = [[] for _ in range(self.num_rounds)]

        for i in range(len(samples)):
            inputs = samples.inputs[i].tolist()
            k, coeffs, rounds = tournament_gcd(inputs)

            assert k == samples.k[i]
            coeffs_list.append(coeffs)

            # Store round annotations
            for r, round_data in enumerate(rounds):
                gcds = [g for g, u, v in round_data]
                us = [u for g, u, v in round_data]
                vs = [v for g, u, v in round_data]
                round_gcds[r].append(gcds)
                round_u[r].append(us)
                round_v[r].append(vs)

        # Stack into arrays
        coeffs = np.stack(coeffs_list, axis=0)
        round_gcds = [np.array(r, dtype=DTYPE) for r in round_gcds]
        round_u = [np.array(r, dtype=DTYPE) for r in round_u]
        round_v = [np.array(r, dtype=DTYPE) for r in round_v]

        return TournamentAnnotatedTranscript(
            self.n, samples.inputs, samples.k, coeffs,
            round_gcds, round_u, round_v
        )
