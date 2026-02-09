"""Label conventions for tournament GCD with n inputs.

This module defines how we name components in the tournament GCD sequence encoding.
The tournament GCD extends the two-input GCD (a, b) → (k, u, v) to n inputs (x₁, ..., xₙ)
→ (k, c₁, ..., cₙ) where sum(cᵢ * xᵢ) = k.

Naming scheme:
- x1, x2, ..., xn: The n input integers (1-indexed for readability)
- k: The final GCD of all inputs
- c1, c2, ..., cn: Bézout coefficients where sum(cᵢ * xᵢ) = k
- g{r}_{p}, u{r}_{p}, v{r}_{p}: Intermediate results at round r, pair p (for annotations)

Example for n = 4:
    Inputs: x1, x2, x3, x4
    Outputs: k, c1, c2, c3, c4
    Annotations (Option A): g0_0, u0_0, v0_0, g0_1, u0_1, v0_1, g1_0, u1_0, v1_0

TournamentLabels provides static methods to generate these labels programmatically.
"""


class TournamentLabels:
    """Labels for tournament GCD sequences.

    Naming convention:
    - x1, x2, ..., xn: inputs
    - k: final GCD
    - c1, c2, ..., cn: Bézout coefficients where sum(ci * xi) = k
    - g{round}_{pair}: intermediate GCD at round r, pair p
    - u{round}_{pair}, v{round}_{pair}: Bézout coefficients for that pair
    """

    k: str = "k"  # Final GCD

    @staticmethod
    def x(i: int) -> str:
        """Input label: x1, x2, ..., xn (1-indexed)."""
        return f"x{i}"

    @staticmethod
    def c(i: int) -> str:
        """Coefficient label: c1, c2, ..., cn (1-indexed)."""
        return f"c{i}"

    @staticmethod
    def g(round: int, pair: int) -> str:
        """Intermediate GCD at round r, pair p."""
        return f"g{round}_{pair}"

    @staticmethod
    def u(round: int, pair: int) -> str:
        """Bézout u coefficient at round r, pair p."""
        return f"u{round}_{pair}"

    @staticmethod
    def v(round: int, pair: int) -> str:
        """Bézout v coefficient at round r, pair p."""
        return f"v{round}_{pair}"

    @classmethod
    def input_labels(cls, n: int) -> list[str]:
        """Return input labels for n inputs: [x1, x2, ..., xn]."""
        return [cls.x(i) for i in range(1, n + 1)]

    @classmethod
    def coefficient_labels(cls, n: int) -> list[str]:
        """Return coefficient labels: [c1, c2, ..., cn]."""
        return [cls.c(i) for i in range(1, n + 1)]

    @classmethod
    def transcript_target_labels(cls, n: int) -> list[str]:
        """Target labels for transcript (no annotations): [k, c1, ..., cn]."""
        return [cls.k] + cls.coefficient_labels(n)

    @classmethod
    def annotation_labels(cls, n: int) -> list[str]:
        """Annotation labels for all rounds (Option A).

        For n inputs, there are log2(n) rounds.
        Round 0: n/2 pairs
        Round 1: n/4 pairs
        ...
        Round log2(n)-1: 1 pair
        """
        import math

        num_rounds = int(math.log2(n))
        labels = []
        for r in range(num_rounds):
            num_pairs = n // (2 ** (r + 1))
            for p in range(num_pairs):
                labels.extend([cls.g(r, p), cls.u(r, p), cls.v(r, p)])
        return labels

    @classmethod
    def annotated_target_labels(cls, n: int) -> list[str]:
        """Target labels for annotated transcript: [annotations..., k, c1, ..., cn]."""
        return cls.annotation_labels(n) + cls.transcript_target_labels(n)
