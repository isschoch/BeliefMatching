#!/usr/bin/env python3
"""
Simple benchmark for the BeliefMatching decoder.

Times decoder construction, single-shot decode, and batch decode at various
batch sizes. Use this to measure baseline performance and to validate
optimizations.

Usage:
    python benchmarks/benchmark_belief_matching.py
    python benchmarks/benchmark_belief_matching.py --distance 5 --distance 7
"""

import argparse
import time
from typing import List, Tuple

import numpy as np
import stim

from beliefmatching import BeliefMatching


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


def time_init(circuit: stim.Circuit, runs: int = 5) -> Tuple[float, float]:
    """Time decoder construction (mean and std over runs, seconds)."""
    times_s = []
    for _ in range(runs):
        t0 = time.perf_counter()
        BeliefMatching(circuit, max_bp_iters=20, bp_method="product_sum")
        times_s.append(time.perf_counter() - t0)
    return float(np.mean(times_s)), float(np.std(times_s))


def time_decode_single(
    bm: BeliefMatching,
    syndromes: np.ndarray,
    warmup: int = 3,
    timed_runs: int = 50,
) -> Tuple[float, float]:
    """Time single-shot decode: mean and std per decode (seconds)."""
    for i in range(warmup):
        _ = bm.decode(syndromes[i % len(syndromes)])
    times_s = []
    for i in range(timed_runs):
        s = syndromes[i % len(syndromes)]
        t0 = time.perf_counter()
        _ = bm.decode(s)
        times_s.append(time.perf_counter() - t0)
    return float(np.mean(times_s)), float(np.std(times_s))


def time_decode_batch(
    bm: BeliefMatching,
    shots: np.ndarray,
    warmup: int = 1,
    timed_runs: int = 5,
) -> Tuple[float, float]:
    """Time batch decode: mean and std per batch (seconds)."""
    for _ in range(warmup):
        _ = bm.decode_batch(shots)
    times_s = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        _ = bm.decode_batch(shots)
        times_s.append(time.perf_counter() - t0)
    return float(np.mean(times_s)), float(np.std(times_s))


def run_benchmark(
    distance: int,
    num_syndrome_shots: int = 200,
    batch_sizes: List[int] = None,
) -> None:
    if batch_sizes is None:
        batch_sizes = [1, 10, 50, 100, 200]

    p = 0.01
    circuit = make_surface_code_circuit(distance, p)
    dem = circuit.detector_error_model(decompose_errors=True)
    n_det = dem.num_detectors
    n_obs = dem.num_observables

    print(f"\n--- Surface code d={distance} (detectors={n_det}, observables={n_obs}) ---")

    # Decoder init
    mean_s, std_s = time_init(circuit)
    print(f"  Decoder init:     {mean_s*1000:.2f} ± {std_s*1000:.2f} ms")

    bm = BeliefMatching(circuit, max_bp_iters=20, bp_method="product_sum")
    syndromes = sample_syndromes(circuit, num_syndrome_shots)

    # Single-shot decode
    mean_s, std_s = time_decode_single(bm, syndromes)
    print(f"  Single decode:    {mean_s*1000:.2f} ± {std_s*1000:.2f} ms  ({1/mean_s:.0f} decodes/s)")

    # Batch decode
    for batch_size in batch_sizes:
        if batch_size > num_syndrome_shots:
            continue
        batch = syndromes[:batch_size]
        mean_s, std_s = time_decode_batch(bm, batch)
        per_shot_ms = mean_s * 1000 / batch_size
        throughput = batch_size / mean_s
        print(f"  Batch ({batch_size:4d}):      {mean_s*1000:.2f} ± {std_s*1000:.2f} ms total  "
              f"{per_shot_ms:.3f} ms/shot  {throughput:.0f} decodes/s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark BeliefMatching decoder")
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
    args = parser.parse_args()

    print("BeliefMatching decoder benchmark")
    print("  (warmup + repeated runs; report mean ± std)")

    for d in args.distance:
        run_benchmark(d, num_syndrome_shots=args.shots, batch_sizes=args.batch_sizes)

    print()


if __name__ == "__main__":
    main()
