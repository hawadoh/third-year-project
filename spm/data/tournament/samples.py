"""Sample classes for tournament GCD with n inputs.

This module defines data containers for tournament GCD samples — batches of (inputs, outputs)
that get encoded into training sequences. There are three levels of data:

1. TournamentSamples: Just inputs and GCD
   - inputs: (N, n) array — N samples of n inputs each
   - k: (N,) array — the GCD of each sample

2. TournamentTranscript: Adds Bézout coefficients
   - coeffs: (N, n) array — coefficients where sum(cᵢ * xᵢ) = k

3. TournamentAnnotatedTranscript: Adds intermediate tournament rounds (Option A from my Progress Report)
   - round_gcds, round_u, round_v: Lists of arrays storing g, u, v for each round
   - For n = 4: round 0 has 2 pairs, round 1 has 1 pair → depth 2

Each class has:
- to_dict(): Converts to {label: array} dictionary for encoding
- remove(indices): Removes samples at given indices
- add(other): Concatenates with another batch

The to_dict() output gets fed to EncodedSamples → TensorRepr → x.bin/y.bin files.
"""

import numpy as np

from spm.data.samples import DTYPE
from spm.data.tournament.labels import TournamentLabels


class TournamentSamples:
    """
    Base class for tournament GCD samples with n inputs.

    Attributes:
        n: Number of inputs (4, 8, or 16)
        inputs: np.ndarray of shape (num_samples, n) - the input values
        k: np.ndarray of shape (num_samples,) - the GCD values
    """

    def __init__(self, n: int, inputs: np.ndarray, k: np.ndarray):
        assert n in [4, 8, 16], f"n must be 4, 8, or 16, got {n}"
        assert inputs.shape[1] == n, f"inputs must have {n} columns"
        assert inputs.shape[0] == k.shape[0], "inputs and k must have same number of samples"

        self.n = n
        self.inputs = inputs  # Shape: (num_samples, n)
        self.k = k            # Shape: (num_samples,)

    def to_dict(self) -> dict:
        """Convert to dictionary with labeled columns."""
        d = {}
        for i in range(self.n):
            d[TournamentLabels.x(i + 1)] = self.inputs[:, i]
        d[TournamentLabels.k] = self.k
        return d

    def __len__(self):
        return self.inputs.shape[0]

    @property
    def a(self):
        """First input for DisjointRejector compatibility."""
        return self.inputs[:, 0]

    @property
    def b(self):
        """Second input for DisjointRejector compatibility."""
        return self.inputs[:, 1]

    def remove(self, indices: list[int]):
        """Remove samples at given indices."""
        self.inputs = np.delete(self.inputs, indices, axis=0)
        self.k = np.delete(self.k, indices, axis=0)

    def add(self, other: "TournamentSamples"):
        """Concatenate with another TournamentSamples."""
        assert self.n == other.n
        self.inputs = np.concatenate((self.inputs, other.inputs), axis=0)
        self.k = np.concatenate((self.k, other.k), axis=0)


class TournamentTranscript(TournamentSamples):
    """
    Tournament GCD samples with Bézout coefficients (no intermediate annotations).

    Additional attributes:
        coeffs: np.ndarray of shape (num_samples, n) - Bézout coefficients
                where sum(coeffs[i] * inputs[i]) = k for each sample
    """

    def __init__(self, n: int, inputs: np.ndarray, k: np.ndarray, coeffs: np.ndarray):
        super().__init__(n, inputs, k)
        assert coeffs.shape == inputs.shape, "coeffs must have same shape as inputs"
        self.coeffs = coeffs  # Shape: (num_samples, n)

    def to_dict(self) -> dict:
        """Convert to dictionary: x1, ..., xn, k, c1, ..., cn."""
        d = super().to_dict()
        for i in range(self.n):
            d[TournamentLabels.c(i + 1)] = self.coeffs[:, i]
        return d

    def remove(self, indices: list[int]):
        """Remove samples at given indices."""
        super().remove(indices)
        self.coeffs = np.delete(self.coeffs, indices, axis=0)

    def add(self, other: "TournamentTranscript"):
        """Concatenate with another TournamentTranscript."""
        super().add(other)
        self.coeffs = np.concatenate((self.coeffs, other.coeffs), axis=0)


class TournamentAnnotatedTranscript(TournamentTranscript):
    """
    Tournament GCD with Option A annotations: intermediate GCD results at each round.

    For n inputs, there are log2(n) rounds.
    Round 0: n/2 pairs -> n/2 intermediate GCDs
    Round 1: n/4 pairs -> n/4 intermediate GCDs
    ...
    Round log2(n)-1: 1 pair -> final GCD

    Additional attributes:
        round_gcds: List of np.ndarray, one per round.
                    round_gcds[r] has shape (num_samples, n // 2^(r+1))
        round_u: List of np.ndarray, Bézout u coefficients per round
        round_v: List of np.ndarray, Bézout v coefficients per round
    """

    def __init__(
        self,
        n: int,
        inputs: np.ndarray,
        k: np.ndarray,
        coeffs: np.ndarray,
        round_gcds: list[np.ndarray],
        round_u: list[np.ndarray],
        round_v: list[np.ndarray],
    ):
        super().__init__(n, inputs, k, coeffs)
        self.num_rounds = int(np.log2(n))
        assert len(round_gcds) == self.num_rounds
        assert len(round_u) == self.num_rounds
        assert len(round_v) == self.num_rounds
        self.round_gcds = round_gcds
        self.round_u = round_u
        self.round_v = round_v

    def to_dict(self) -> dict:
        """
        Convert to dictionary with annotations.

        Order: x1, ..., xn, [round 0 annotations], [round 1 annotations], ..., k, c1, ..., cn

        Round r annotations: g_r_0, u_r_0, v_r_0, g_r_1, u_r_1, v_r_1, ...
        """
        d = {}

        # Inputs first
        for i in range(self.n):
            d[TournamentLabels.x(i + 1)] = self.inputs[:, i]

        # Annotations per round
        for r in range(self.num_rounds):
            num_pairs = self.n // (2 ** (r + 1))
            for p in range(num_pairs):
                d[TournamentLabels.g(r, p)] = self.round_gcds[r][:, p]
                d[TournamentLabels.u(r, p)] = self.round_u[r][:, p]
                d[TournamentLabels.v(r, p)] = self.round_v[r][:, p]

        # Final outputs
        d[TournamentLabels.k] = self.k
        for i in range(self.n):
            d[TournamentLabels.c(i + 1)] = self.coeffs[:, i]

        return d

    def remove(self, indices: list[int]):
        """Remove samples at given indices."""
        super().remove(indices)
        for r in range(self.num_rounds):
            self.round_gcds[r] = np.delete(self.round_gcds[r], indices, axis=0)
            self.round_u[r] = np.delete(self.round_u[r], indices, axis=0)
            self.round_v[r] = np.delete(self.round_v[r], indices, axis=0)

    def add(self, other: "TournamentAnnotatedTranscript"):
        """Concatenate with another TournamentAnnotatedTranscript."""
        super().add(other)
        for r in range(self.num_rounds):
            self.round_gcds[r] = np.concatenate((self.round_gcds[r], other.round_gcds[r]), axis=0)
            self.round_u[r] = np.concatenate((self.round_u[r], other.round_u[r]), axis=0)
            self.round_v[r] = np.concatenate((self.round_v[r], other.round_v[r]), axis=0)
