"""Sample class for tournament GCD Option B annotations.

Option B extends Option A by storing the full Euclidean algorithm trace within each
pairwise GCD computation, rather than just the final (g, u, v) result.

For each pair (r, p) in the tournament tree:
    - q_traces[r][p]: shape (num_samples, T)  — quotients at each step
    - u_traces[r][p]: shape (num_samples, T)  — running Bézout u at each step
    - v_traces[r][p]: shape (num_samples, T)  — running Bézout v at each step
    - round_gcds[r][p]: shape (num_samples,)  — final GCD of this pair
    - round_u[r][p]:   shape (num_samples,)   — final Bézout u of this pair
    - round_v[r][p]:   shape (num_samples,)   — final Bézout v of this pair

Shorter Euclidean sequences are padded by repeating the last element (edge padding),
consistent with the n=2 AnnotatedTranscriptSampler.
"""

import numpy as np

from spm.data.samples import DTYPE
from spm.data.tournament_optionb.labels import TournamentOptionBLabels


class TournamentOptionBSamples:
    """Tournament GCD samples with full Option B Euclidean traces per pair.

    Attributes:
        n:          Number of inputs (must be power of 2: 4, 8, or 16)
        T:          Number of Euclidean trace steps per pair
        inputs:     (num_samples, n) — input integers
        k:          (num_samples,)   — overall GCD
        coeffs:     (num_samples, n) — final Bézout coefficients where sum(ci * xi) = k
        q_traces:   list of lists — q_traces[r][p] has shape (num_samples, T)
        u_traces:   list of lists — u_traces[r][p] has shape (num_samples, T)
        v_traces:   list of lists — v_traces[r][p] has shape (num_samples, T)
        round_gcds: list of np.ndarray — round_gcds[r] has shape (num_samples, n//2^(r+1))
        round_u:    list of np.ndarray — final Bézout u per pair per round
        round_v:    list of np.ndarray — final Bézout v per pair per round
    """

    def __init__(
        self,
        n: int,
        T: int,
        inputs: np.ndarray,
        k: np.ndarray,
        coeffs: np.ndarray,
        q_traces: list,
        u_traces: list,
        v_traces: list,
        round_gcds: list,
        round_u: list,
        round_v: list,
    ):
        assert n in [4, 8, 16]
        self.n = n
        self.T = T
        self.num_rounds = int(np.log2(n))
        self.inputs = inputs
        self.k = k
        self.coeffs = coeffs
        # q_traces[r][p], u_traces[r][p], v_traces[r][p]: shape (num_samples, T)
        self.q_traces = q_traces
        self.u_traces = u_traces
        self.v_traces = v_traces
        # round_gcds[r], round_u[r], round_v[r]: shape (num_samples, num_pairs_in_round)
        self.round_gcds = round_gcds
        self.round_u = round_u
        self.round_v = round_v

    def __len__(self):
        return self.inputs.shape[0]

    @property
    def a(self):
        """DisjointRejector compatibility."""
        return self.inputs[:, 0]

    @property
    def b(self):
        """DisjointRejector compatibility."""
        return self.inputs[:, 1]

    def to_dict(self) -> dict:
        """Convert to {label: array} dictionary.

        Target order: k, [for each pair: trace steps then final g/u/v], c1, ..., cn
        This is consistent with TournamentOptionBLabels.target_labels(n, T).
        """
        d = {}

        # Inputs
        for i in range(self.n):
            d[TournamentOptionBLabels.x(i + 1)] = self.inputs[:, i]

        # k first
        d[TournamentOptionBLabels.k] = self.k

        # Annotations: for each round and pair, trace steps then final (g, u, v)
        for r in range(self.num_rounds):
            num_pairs = self.n // (2 ** (r + 1))
            for p in range(num_pairs):
                # Trace steps: u, v, q per step (consistent with n=2 ATL convention)
                for step in range(self.T):
                    d[TournamentOptionBLabels.u_step(r, p, step)] = self.u_traces[r][p][:, step]
                    d[TournamentOptionBLabels.v_step(r, p, step)] = self.v_traces[r][p][:, step]
                    d[TournamentOptionBLabels.q_step(r, p, step)] = self.q_traces[r][p][:, step]
                # Final (g, u, v) for this pair
                d[TournamentOptionBLabels.g(r, p)] = self.round_gcds[r][:, p]
                d[TournamentOptionBLabels.u_final(r, p)] = self.round_u[r][:, p]
                d[TournamentOptionBLabels.v_final(r, p)] = self.round_v[r][:, p]

        # Final Bézout coefficients
        for i in range(self.n):
            d[TournamentOptionBLabels.c(i + 1)] = self.coeffs[:, i]

        return d

    def remove(self, indices: list[int]):
        self.inputs = np.delete(self.inputs, indices, axis=0)
        self.k = np.delete(self.k, indices, axis=0)
        self.coeffs = np.delete(self.coeffs, indices, axis=0)
        for r in range(self.num_rounds):
            num_pairs = self.n // (2 ** (r + 1))
            for p in range(num_pairs):
                self.q_traces[r][p] = np.delete(self.q_traces[r][p], indices, axis=0)
                self.u_traces[r][p] = np.delete(self.u_traces[r][p], indices, axis=0)
                self.v_traces[r][p] = np.delete(self.v_traces[r][p], indices, axis=0)
            self.round_gcds[r] = np.delete(self.round_gcds[r], indices, axis=0)
            self.round_u[r] = np.delete(self.round_u[r], indices, axis=0)
            self.round_v[r] = np.delete(self.round_v[r], indices, axis=0)

    def add(self, other: "TournamentOptionBSamples"):
        assert self.n == other.n and self.T == other.T
        self.inputs = np.concatenate((self.inputs, other.inputs), axis=0)
        self.k = np.concatenate((self.k, other.k), axis=0)
        self.coeffs = np.concatenate((self.coeffs, other.coeffs), axis=0)
        for r in range(self.num_rounds):
            num_pairs = self.n // (2 ** (r + 1))
            for p in range(num_pairs):
                self.q_traces[r][p] = np.concatenate((self.q_traces[r][p], other.q_traces[r][p]), axis=0)
                self.u_traces[r][p] = np.concatenate((self.u_traces[r][p], other.u_traces[r][p]), axis=0)
                self.v_traces[r][p] = np.concatenate((self.v_traces[r][p], other.v_traces[r][p]), axis=0)
            self.round_gcds[r] = np.concatenate((self.round_gcds[r], other.round_gcds[r]), axis=0)
            self.round_u[r] = np.concatenate((self.round_u[r], other.round_u[r]), axis=0)
            self.round_v[r] = np.concatenate((self.round_v[r], other.round_v[r]), axis=0)
