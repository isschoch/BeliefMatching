#!/usr/bin/env python3
"""
Benchmark for BeliefMatching and (optionally) PyMatching (MWPM).

Times decoder construction, single-shot decode, and batch decode at various
batch sizes for both decoders for direct comparison.

Usage:
    python benchmarks/benchmark_belief_matching.py
    python benchmarks/benchmark_belief_matching.py --distance 5 7 --shots 500
    python benchmarks/benchmark_belief_matching.py --no-compare-mwpm
"""

import argparse
import time
from typing import List, Tuple, Optional

import numpy as np
import stim

from beliefmatching import BeliefMatching

try:
    import pymatching

    PYMATCHING_AVAILABLE = True
except Exception:
    PYMATCHING_AVAILABLE = False


def make_surface_code_circuit(distance: int, p: float = 0.01) -> stim.Circuit:
    """Generate a rotated surface code circuit (no I/O)."""
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        rounds=distance,
        distance=distance,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        after_clifford_depolarization=p,
    )


def sample_syndromes(circuit: stim.Circuit, num_shots: int) -> np.ndarray:
    """Sample random syndromes (detector outcomes only)."""
    sampler = circuit.compile_detector_sampler()
    shots, _ = sampler.sample(num_shots, separate_observables=True)
    return shots  # shape (num_shots, num_detectors)


def time_init_beliefmatching(
    circuit: stim.Circuit, runs: int = 5
) -> Tuple[float, float]:
    """Time BeliefMatching constructor (mean and std over runs, seconds)."""
    times_s = []
    for _ in range(runs):
        t0 = time.perf_counter()
        BeliefMatching(circuit, max_bp_iters=20, bp_method="product_sum")
        times_s.append(time.perf_counter() - t0)
    return float(np.mean(times_s)), float(np.std(times_s))


def time_init_pymatching(
    dem: stim.DetectorErrorModel, runs: int = 5
) -> Tuple[float, float]:
    """Time PyMatching Matching initialization (mean and std over runs, seconds)."""
    if not PYMATCHING_AVAILABLE:
        return float("nan"), float("nan")
    times_s = []
    for _ in range(runs):
        t0 = time.perf_counter()
        # Construct Matching directly from stim DetectorErrorModel.
        _ = pymatching.Matching(dem)
        times_s.append(time.perf_counter() - t0)
    return float(np.mean(times_s)), float(np.std(times_s))


def time_decode_single_beliefmatching(
    bm: BeliefMatching,
    syndromes: np.ndarray,
    warmup: int = 3,
    timed_runs: int = 50,
) -> Tuple[float, float]:
    """Time single-shot decode for BeliefMatching: mean and std per decode (seconds)."""
    for i in range(warmup):
        _ = bm.decode(syndromes[i % len(syndromes)])
    times_s = []
    for i in range(timed_runs):
        s = syndromes[i % len(syndromes)]
        t0 = time.perf_counter()
        _ = bm.decode(s)
        times_s.append(time.perf_counter() - t0)
    return float(np.mean(times_s)), float(np.std(times_s))


def time_decode_batch_beliefmatching(
    bm: BeliefMatching,
    shots: np.ndarray,
    warmup: int = 1,
    timed_runs: int = 5,
) -> Tuple[float, float]:
    """Time batch decode for BeliefMatching: mean and std per batch (seconds)."""
    for _ in range(warmup):
        _ = bm.decode_batch(shots)
    times_s = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        _ = bm.decode_batch(shots)
        times_s.append(time.perf_counter() - t0)
    return float(np.mean(times_s)), float(np.std(times_s))


def time_decode_single_pymatching(
    mm: "pymatching.Matching",
    syndromes: np.ndarray,
    warmup: int = 3,
    timed_runs: int = 50,
) -> Tuple[float, float]:
    """Time single-shot decode for PyMatching: mean and std per decode (seconds)."""
    if not PYMATCHING_AVAILABLE:
        return float("nan"), float("nan")
    # ensure syndromes are integer/bool 1D arrays when passed to decode
    for i in range(warmup):
        _ = mm.decode(syndromes[i % len(syndromes)])
    times_s = []
    for i in range(timed_runs):
        s = syndromes[i % len(syndromes)]
        t0 = time.perf_counter()
        _ = mm.decode(s)
        times_s.append(time.perf_counter() - t0)
    return float(np.mean(times_s)), float(np.std(times_s))


def time_decode_batch_pymatching(
    mm: "pymatching.Matching",
    shots: np.ndarray,
    warmup: int = 1,
    timed_runs: int = 5,
) -> Tuple[float, float]:
    """Time batch decode for PyMatching: mean and std per batch (seconds)."""
    if not PYMATCHING_AVAILABLE:
        return float("nan"), float("nan")
    for _ in range(warmup):
        _ = mm.decode_batch(shots)
    times_s = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        _ = mm.decode_batch(shots)
        times_s.append(time.perf_counter() - t0)
    return float(np.mean(times_s)), float(np.std(times_s))


def run_benchmark(
    distance: int,
    num_syndrome_shots: int = 200,
    batch_sizes: List[int] = None,
    compare_mwpm: bool = True,
) -> None:
    if batch_sizes is None:
        batch_sizes = [1, 10, 50, 100, 200]

    p = 0.01
    circuit = make_surface_code_circuit(distance, p)
    dem = circuit.detector_error_model(decompose_errors=True)
    n_det = dem.num_detectors
    n_obs = dem.num_observables

    print(
        f"\n--- Surface code d={distance} (detectors={n_det}, observables={n_obs}) ---"
    )

    # ----------------
    # BeliefMatching
    # ----------------
    mean_s, std_s = time_init_beliefmatching(circuit)
    print(f"BeliefMatching:")
    print(f"  Decoder init:     {mean_s*1000:.2f} ± {std_s*1000:.2f} ms")

    bm = BeliefMatching(circuit, max_bp_iters=20, bp_method="product_sum")
    syndromes = sample_syndromes(circuit, num_syndrome_shots)

    mean_s, std_s = time_decode_single_beliefmatching(bm, syndromes)
    print(
        f"  Single decode:    {mean_s*1000:.2f} ± {std_s*1000:.2f} ms  ({0 if mean_s==0 else 1/mean_s:.0f} decodes/s)"
    )

    for batch_size in batch_sizes:
        if batch_size > num_syndrome_shots:
            continue
        batch = syndromes[:batch_size]
        mean_s, std_s = time_decode_batch_beliefmatching(bm, batch)
        per_shot_ms = mean_s * 1000 / batch_size
        throughput = batch_size / mean_s if mean_s != 0 else float("inf")
        print(
            f"  Batch ({batch_size:4d}):      {mean_s*1000:.2f} ± {std_s*1000:.2f} ms total  "
            f"{per_shot_ms:.3f} ms/shot  {throughput:.0f} decodes/s"
        )

    # ----------------
    # PyMatching (MWPM)
    # ----------------
    if compare_mwpm:
        print("\nPyMatching (MWPM):")
        if not PYMATCHING_AVAILABLE:
            print(
                "  pymatching not available (import failed). Install with 'pip install pymatching' to enable."
            )
            return

        mean_s, std_s = time_init_pymatching(dem)
        print(f"  Decoder init:     {mean_s*1000:.2f} ± {std_s*1000:.2f} ms")

        # create a single persistent Matching to decode with
        mm = pymatching.Matching(dem)

        mean_s, std_s = time_decode_single_pymatching(mm, syndromes)
        print(
            f"  Single decode:    {mean_s*1000:.2f} ± {std_s*1000:.2f} ms  ({0 if mean_s==0 else 1/mean_s:.0f} decodes/s)"
        )

        for batch_size in batch_sizes:
            if batch_size > num_syndrome_shots:
                continue
            batch = syndromes[:batch_size]
            mean_s, std_s = time_decode_batch_pymatching(mm, batch)
            per_shot_ms = mean_s * 1000 / batch_size
            throughput = batch_size / mean_s if mean_s != 0 else float("inf")
            print(
                f"  Batch ({batch_size:4d}):      {mean_s*1000:.2f} ± {std_s*1000:.2f} ms total  "
                f"{per_shot_ms:.3f} ms/shot  {throughput:.0f} decodes/s"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark BeliefMatching and optional PyMatching decoder"
    )
    parser.add_argument(
        "--distance",
        type=int,
        default=[7],
        nargs="+",
        help="Code distances to benchmark (default: 7)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=200,
        help="Number of syndrome samples for timing (default: 200)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 10, 50, 100, 200],
        help="Batch sizes for decode_batch (default: 1 10 50 100 200)",
    )
    parser.add_argument(
        "--compare-mwpm",
        dest="compare_mwpm",
        action="store_true",
        help="Don't benchmark PyMatching (MWPM)",
    )
    args = parser.parse_args()

    print("BeliefMatching vs PyMatching (MWPM) benchmark")
    print("  (warmup + repeated runs; report mean ± std)")

    for d in args.distance:
        run_benchmark(
            d,
            num_syndrome_shots=args.shots,
            batch_sizes=args.batch_sizes,
            compare_mwpm=args.compare_mwpm,
        )

    print()


if __name__ == "__main__":
    main()
