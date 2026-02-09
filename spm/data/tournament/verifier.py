"""Linear verifier for tournament GCD — O(n) time complexity.

This module implements the verifier component described in Progress Report Section 4.3.
The verifier checks whether a claimed GCD k and Bézout coefficients (c₁, ..., cₙ) are correct
for given inputs (x₁, ..., xₙ) in linear time.

Two checks (both O(n)):
1. Bézout's identity: sum(cᵢ * xᵢ) = k
   → Confirms k is a linear combination of the inputs

2. Common divisor: k divides all xᵢ
   → Combined with (1), proves k is the GCD

Usage:
    >>> from spm.data.tournament.verifier import verify
    >>> verify([12, 18, 8, 24], 2, [1, -1, 1, 0])
    True

This is the manual verifier for ground-truth validation. During model training,
the model learns to generate (k, c₁, ..., cₙ) that pass this verification.
"""

from typing import List


def verify(inputs: List[int], k: int, coeffs: List[int]) -> bool:
    """
    Verify that k is the GCD of inputs and coeffs are valid Bézout coefficients.

    This is the linear verifier described in Section 4.3 of my Progress Report.
    It performs two checks in O(n) time:

    1. Bézout's identity: sum(c_i * x_i) = k
       This confirms k is a linear combination of the inputs.

    2. Common divisor: k divides all x_i
       Combined with Check 1, this proves k is the GCD.

    Args:
        inputs: List of n input integers [x1, x2, ..., xn]
        k: Claimed GCD value
        coeffs: List of n Bézout coefficients [c1, c2, ..., cn]

    Returns:
        True if both checks pass, False otherwise.

    Example:
        >>> verify([12, 18, 8, 24], 2, [1, -1, 1, 0])
        True
        >>> # 1*12 + (-1)*18 + 1*8 + 0*24 = 12 - 18 + 8 = 2 ✓
        >>> # 2 divides 12, 18, 8, 24 ✓
    """
    assert len(inputs) == len(coeffs), "inputs and coeffs must have same length"

    # Check 1: Bézout's identity
    bezout_sum = sum(c * x for c, x in zip(coeffs, inputs))
    if bezout_sum != k:
        return False

    # Check 2: k divides all inputs
    for x in inputs:
        if x % k != 0:
            return False

    return True
