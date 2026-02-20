"""Label conventions for tournament GCD Option B annotations.

Option B shows the full Euclidean algorithm trace within each pairwise GCD computation,
rather than just the final (g, u, v) result as in Option A.

For each tournament tree node (round r, pair p), with up to T trace steps:
    - Step i (i=0..T-1): q{r}_{p}_{i}, u{r}_{p}_{i}, v{r}_{p}_{i}
      (quotient and running Bézout coefficients at each Euclidean step)
    - Final: g{r}_{p}, u{r}_{p}, v{r}_{p}
      (the GCD of this pair and its final Bézout coefficients)

All annotation labels contain underscores, so the existing TRANSCRIPT masking logic in
tensor_repr.py correctly identifies them as annotations to exclude.

Example for n=4, T=7 (3 pairs total: 2 in round 0, 1 in round 1):
    Target sequence: k, [trace pair 0-0], [trace pair 0-1], [trace pair 1-0], c1, c2, c3, c4
    Per pair: q0_0_0 u0_0_0 v0_0_0 ... q0_0_6 u0_0_6 v0_0_6 g0_0 u0_0 v0_0
    Total annotation labels: 3 pairs * (7*3 + 3) = 72
"""

import math


class TournamentOptionBLabels:
    """Labels for tournament GCD Option B sequences."""

    k: str = "k"

    @staticmethod
    def x(i: int) -> str:
        """Input label: x1, x2, ..., xn (1-indexed)."""
        return f"x{i}"

    @staticmethod
    def c(i: int) -> str:
        """Final Bézout coefficient: c1, c2, ..., cn (1-indexed)."""
        return f"c{i}"

    @staticmethod
    def g(r: int, p: int) -> str:
        """Final GCD of pair p in round r."""
        return f"g{r}_{p}"

    @staticmethod
    def u_final(r: int, p: int) -> str:
        """Final Bézout u for pair p in round r (same position as Option A u{r}_{p})."""
        return f"u{r}_{p}"

    @staticmethod
    def v_final(r: int, p: int) -> str:
        """Final Bézout v for pair p in round r (same position as Option A v{r}_{p})."""
        return f"v{r}_{p}"

    @staticmethod
    def q_step(r: int, p: int, step: int) -> str:
        """Quotient at step `step` of the Euclidean algorithm for pair p in round r."""
        return f"q{r}_{p}_{step}"

    @staticmethod
    def u_step(r: int, p: int, step: int) -> str:
        """Running Bézout u at step `step` for pair p in round r."""
        return f"u{r}_{p}_{step}"

    @staticmethod
    def v_step(r: int, p: int, step: int) -> str:
        """Running Bézout v at step `step` for pair p in round r."""
        return f"v{r}_{p}_{step}"

    @classmethod
    def input_labels(cls, n: int) -> list[str]:
        return [cls.x(i) for i in range(1, n + 1)]

    @classmethod
    def coefficient_labels(cls, n: int) -> list[str]:
        return [cls.c(i) for i in range(1, n + 1)]

    @classmethod
    def pair_annotation_labels(cls, r: int, p: int, T: int) -> list[str]:
        """Labels for a single pair (r, p) with T trace steps.

        Order mirrors n=2 ATL: u, v, q per step, then final g, u, v.
        u0, v0, q0, u1, v1, q1, ..., u_{T-1}, v_{T-1}, q_{T-1}, g, u_final, v_final
        """
        labels = []
        for step in range(T):
            labels.extend([cls.u_step(r, p, step), cls.v_step(r, p, step), cls.q_step(r, p, step)])
        labels.extend([cls.g(r, p), cls.u_final(r, p), cls.v_final(r, p)])
        return labels

    @classmethod
    def annotation_labels(cls, n: int, T: int) -> list[str]:
        """All annotation labels for n inputs with T trace steps per pair."""
        num_rounds = int(math.log2(n))
        labels = []
        for r in range(num_rounds):
            num_pairs = n // (2 ** (r + 1))
            for p in range(num_pairs):
                labels.extend(cls.pair_annotation_labels(r, p, T))
        return labels

    @classmethod
    def target_labels(cls, n: int, T: int) -> list[str]:
        """Full target sequence: k, [all annotations], c1, ..., cn."""
        return [cls.k] + cls.annotation_labels(n, T) + cls.coefficient_labels(n)
