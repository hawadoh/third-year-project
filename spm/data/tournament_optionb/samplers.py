"""Sampler for tournament GCD Option B: full Euclidean traces per pairwise GCD.

The sampler runs the tournament GCD algorithm (same as Option A) but additionally
records the full Euclidean algorithm trace for each pairwise GCD computation.

The chain computation and padding logic mirrors AnnotatedTranscriptSampler in
spm/data/samplers.py exactly, but applied per tournament node instead of globally.

Each trace is padded to T steps using edge padding (repeating the last element).
"""

import numpy as np

from spm.data.samples import DTYPE, fcast
from spm.data.tournament_optionb.samples import TournamentOptionBSamples


def egcd_chains(a: int, b: int) -> tuple[list, list, list]:
    """Compute full Euclidean algorithm chains for egcd(a, b).

    Mirrors AnnotatedTranscriptSampler.chains() in spm/data/samplers.py.

    Returns:
        u_chain: running Bézout u coefficients (final u at index [-1])
        v_chain: intermediate remainders + final Bézout v at index [-1]
        q_chain: quotients with dummy leading 0 at index [0]

    The last element of u_chain and v_chain are the final Bézout coefficients:
        u_chain[-1] * a + v_chain[-1] * b = gcd(a, b)
    """
    r, s, q = [a, b], [1, 0], [0]  # q[0] = dummy 0 (same as AnnotatedTranscriptSampler)
    while r[-1] != 0:
        q.append(r[-2] // r[-1])
        r.append(r[-2] - q[-1] * r[-1])
        s.append(s[-2] - q[-1] * s[-1])
    r, s = r[:-1], s[:-1]  # remove trailing zero step
    bezout_v = (r[-1] - a * s[-1]) // b
    # u_chain = s (includes final u at end)
    # v_chain = r[:-1] + [bezout_v]  (intermediate remainders + final v)
    # q_chain = q  (dummy 0 at front)
    return list(s), r[:-1] + [bezout_v], list(q)


def pad_edge(lst: list, length: int) -> np.ndarray:
    """Pad list to `length` using edge padding (repeat last element)."""
    arr = np.array(lst, dtype=DTYPE)
    if len(arr) >= length:
        return arr[:length]
    return np.pad(arr, (0, length - len(arr)), mode="edge")


def process_pair_chains(a: int, b: int, T: int):
    """Run egcd_chains and split into trace + final values.

    The true final Bézout coefficients are always taken from the end of the full
    chain, regardless of T. T only controls how many intermediate steps are stored
    in the trace (shorter chains are padded, longer chains are truncated).

    Returns:
        g:       GCD of (a, b)
        u_final: final Bézout u coefficient  (u_final * a + v_final * b = g)
        v_final: final Bézout v coefficient
        q_trace: np.ndarray of shape (T,) — T quotients (dummy 0 removed)
        u_trace: np.ndarray of shape (T,) — T intermediate u values
        v_trace: np.ndarray of shape (T,) — T intermediate v values
    """
    u_raw, v_raw, q_raw = egcd_chains(a, b)

    # Always use the true final Bézout coefficients from the full chain
    u_final = int(u_raw[-1])
    v_final = int(v_raw[-1])
    g = u_final * a + v_final * b

    # Trace: intermediate steps only (excluding the final Bézout coefficients)
    # Truncate to T if longer, pad with edge if shorter
    u_trace = pad_edge(u_raw[:-1], T)   # u_raw[:-1] = intermediate u values
    v_trace = pad_edge(v_raw[:-1], T)   # v_raw[:-1] = intermediate v values
    q_trace = pad_edge(q_raw[1:], T)    # q_raw[1:]  = quotients without dummy 0

    return g, u_final, v_final, q_trace, u_trace, v_trace


def tournament_gcd_optionb(inputs: list[int], T: int):
    """Run tournament GCD and record full Euclidean traces per pair.

    Args:
        inputs: List of n integers (n must be a power of 2)
        T:      Euclidean trace steps per pair (shorter traces are padded)

    Returns:
        gcd:    Overall GCD of all inputs
        coeffs: Final Bézout coefficients (np.ndarray of shape (n,))
        rounds: List of rounds. Each round is a list of per-pair tuples:
                (g, u_final, v_final, q_trace, u_trace, v_trace)
                where traces are np.ndarray of shape (T,).
    """
    n = len(inputs)

    # Initialise leaf nodes: (value, one-hot Bézout coefficient vector)
    layer = []
    for i, x in enumerate(inputs):
        one_hot = np.zeros(n, dtype=DTYPE)
        one_hot[i] = 1
        layer.append((x, one_hot))

    rounds = []

    while len(layer) > 1:
        next_layer = []
        round_results = []

        for j in range(0, len(layer), 2):
            left_val, left_coeffs = layer[j]
            right_val, right_coeffs = layer[j + 1]

            g, u_fin, v_fin, q_trace, u_trace, v_trace = process_pair_chains(left_val, right_val, T)

            round_results.append((g, u_fin, v_fin, q_trace, u_trace, v_trace))

            # Combine Bézout coefficients upward through the tournament tree
            new_coeffs = u_fin * left_coeffs + v_fin * right_coeffs
            next_layer.append((g, new_coeffs))

        rounds.append(round_results)
        layer = next_layer

    final_gcd, final_coeffs = layer[0]
    return final_gcd, final_coeffs, rounds


class LogUniformTournamentGCDSampler:
    """Sample n integers from 1 to ubound with log-uniform distribution.

    If p_biased > 0, that fraction of samples are generated by first sampling k
    log-uniformly from [1, sqrt(ubound)], then constructing all n inputs as
    k-multiples with coprime quotients (rejection rate ~7.6%). This prevents
    k=1 collapse without giving the model a shortcut from any single pair.
    """

    def __init__(self, n: int, ubound: int, base: int = 10, p_biased: float = 0.0):
        assert n in [4, 8, 16]
        self.n = n
        self.ubound = ubound
        self.base = base
        self.p_biased = p_biased

    def sample_inputs(self, num_samples: int) -> np.ndarray:
        n_biased = int(round(num_samples * self.p_biased))
        n_natural = num_samples - n_biased

        parts = []
        if n_natural > 0:
            parts.append(self._sample_natural(n_natural))
        if n_biased > 0:
            parts.append(self._sample_biased(n_biased))

        inputs = np.vstack(parts) if len(parts) > 1 else parts[0]
        idx = np.random.permutation(num_samples)
        return inputs[idx]

    def _sample_natural(self, num_samples: int) -> np.ndarray:
        log_ubound = np.emath.logn(self.base, self.ubound)
        log_inputs = np.random.uniform(0, log_ubound, size=(num_samples, self.n))
        inputs = np.round(np.power(self.base, log_inputs)).astype(DTYPE, casting="unsafe")
        return np.maximum(inputs, 1)

    def _sample_biased(self, num_samples: int) -> np.ndarray:
        """Sample inputs where k is chosen first, all inputs are k-multiples.

        k is sampled log-uniformly from [1, sqrt(ubound)] so that the quotients
        aᵢ = xᵢ/k have a meaningful range. Samples where gcd(a₁,...,aₙ) ≠ 1
        are rejected (~7.6% rejection rate), ensuring the overall GCD is exactly k.
        """
        sqrt_ubound = max(int(self.ubound ** 0.5), 2)
        log_k_bound = np.emath.logn(self.base, sqrt_ubound)

        # Oversample by ~15% to absorb the rejection step in one vectorised pass
        n_try = int(num_samples * 1.15) + 20

        log_k = np.random.uniform(0, log_k_bound, size=n_try)
        k = np.maximum(np.round(np.power(self.base, log_k)).astype(DTYPE, casting="unsafe"), 1)

        # Per-sample upper bound for each quotient: floor(ubound / k), at least 2
        a_ubound = np.maximum(self.ubound // k, 2).astype(float)
        log_a_bound = np.emath.logn(self.base, a_ubound)
        log_a = np.random.uniform(0, 1, size=(n_try, self.n)) * log_a_bound[:, np.newaxis]
        a = np.maximum(np.round(np.power(self.base, log_a)).astype(DTYPE, casting="unsafe"), 1)

        valid = np.gcd.reduce(a, axis=1) == 1
        inputs = (a[valid] * k[valid, np.newaxis]).astype(DTYPE)

        if len(inputs) >= num_samples:
            return inputs[:num_samples]
        # Rare fallback: didn't get enough after oversampling
        extra = self._sample_biased(num_samples - len(inputs))
        return np.vstack([inputs, extra])

    def __call__(self, num_samples: int) -> np.ndarray:
        return self.sample_inputs(num_samples)


class TournamentOptionBSampler:
    """Sampler for Option B: produces TournamentOptionBSamples with Euclidean traces.

    Args:
        n:       Number of inputs (4, 8, or 16)
        ubound:  Upper bound for input sampling
        T:       Euclidean trace length per pair (default 7, matching n=2 ATL7)
        base:    Base for log-uniform sampling (default 10)
    """

    def __init__(self, n: int, ubound: int, T: int = 7, base: int = 10, p_biased: float = 0.0):
        self.n = n
        self.T = T
        self.num_rounds = int(np.log2(n))
        self.input_sampler = LogUniformTournamentGCDSampler(n, ubound, base, p_biased=p_biased)

    def sample(self, num_samples: int) -> TournamentOptionBSamples:
        inputs = self.input_sampler(num_samples)
        k_arr = np.gcd.reduce(inputs, axis=1)

        coeffs_list = []

        # Per-pair storage: traces[r][p] = list of arrays of shape (T,)
        q_traces = [[[] for _ in range(self.n // (2 ** (r + 1)))] for r in range(self.num_rounds)]
        u_traces = [[[] for _ in range(self.n // (2 ** (r + 1)))] for r in range(self.num_rounds)]
        v_traces = [[[] for _ in range(self.n // (2 ** (r + 1)))] for r in range(self.num_rounds)]
        round_gcds_lists = [[[] for _ in range(self.n // (2 ** (r + 1)))] for r in range(self.num_rounds)]
        round_u_lists    = [[[] for _ in range(self.n // (2 ** (r + 1)))] for r in range(self.num_rounds)]
        round_v_lists    = [[[] for _ in range(self.n // (2 ** (r + 1)))] for r in range(self.num_rounds)]

        for i in range(num_samples):
            sample_inputs = inputs[i].tolist()
            gcd, coeffs, rounds = tournament_gcd_optionb(sample_inputs, self.T)
            assert gcd == k_arr[i], f"GCD mismatch: {gcd} != {k_arr[i]}"

            coeffs_list.append(coeffs)
            for r, round_data in enumerate(rounds):
                for p, (g, u_fin, v_fin, q_trace, u_trace, v_trace) in enumerate(round_data):
                    q_traces[r][p].append(q_trace)
                    u_traces[r][p].append(u_trace)
                    v_traces[r][p].append(v_trace)
                    round_gcds_lists[r][p].append(g)
                    round_u_lists[r][p].append(u_fin)
                    round_v_lists[r][p].append(v_fin)

        # Stack into arrays
        coeffs_arr = np.stack(coeffs_list, axis=0)

        # q/u/v_traces[r][p]: shape (num_samples, T)
        q_traces_arr = [[np.stack(q_traces[r][p], axis=0) for p in range(self.n // (2 ** (r + 1)))]
                        for r in range(self.num_rounds)]
        u_traces_arr = [[np.stack(u_traces[r][p], axis=0) for p in range(self.n // (2 ** (r + 1)))]
                        for r in range(self.num_rounds)]
        v_traces_arr = [[np.stack(v_traces[r][p], axis=0) for p in range(self.n // (2 ** (r + 1)))]
                        for r in range(self.num_rounds)]

        # round_gcds[r]: shape (num_samples, num_pairs_in_round)
        round_gcds = [np.stack(
            [np.array(round_gcds_lists[r][p], dtype=DTYPE) for p in range(self.n // (2 ** (r + 1)))],
            axis=1
        ) for r in range(self.num_rounds)]
        round_u = [np.stack(
            [np.array(round_u_lists[r][p], dtype=DTYPE) for p in range(self.n // (2 ** (r + 1)))],
            axis=1
        ) for r in range(self.num_rounds)]
        round_v = [np.stack(
            [np.array(round_v_lists[r][p], dtype=DTYPE) for p in range(self.n // (2 ** (r + 1)))],
            axis=1
        ) for r in range(self.num_rounds)]

        return TournamentOptionBSamples(
            n=self.n,
            T=self.T,
            inputs=inputs,
            k=k_arr,
            coeffs=coeffs_arr,
            q_traces=q_traces_arr,
            u_traces=u_traces_arr,
            v_traces=v_traces_arr,
            round_gcds=round_gcds,
            round_u=round_u,
            round_v=round_v,
        )

    def __call__(self, num_samples: int) -> TournamentOptionBSamples:
        return self.sample(num_samples)
