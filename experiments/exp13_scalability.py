"""
Experiment 13: Scalability Analysis for Nature Methods

This experiment tests how the inverse prediction method scales with dataset size.
Critical for demonstrating applicability to large spatial transcriptomics datasets.

Measures:
- Training time (with GPU synchronization if available)
- Inference time
- Peak memory usage (CPU and GPU)
- AUC performance at each scale

Scales tested: n_spots = [1000, 3000, 5000, 10000, 20000, 30000]
"""

import gc
import time
import traceback
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

from config import ExperimentConfig, DEFAULT_CONFIG
from core import (
    generate_synthetic_data,
    prepare_data,
    InversePredictionModel,
    train_model,
    compute_scores,
    compute_auc_metrics,
    set_seed,
)


# Cache for CUDA usability check
_cuda_usable = None


def is_cuda_usable() -> bool:
    """Check if CUDA is actually usable (not just detected)."""
    global _cuda_usable
    if _cuda_usable is not None:
        return _cuda_usable

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        _cuda_usable = False
        return False

    try:
        # Try a simple CUDA operation to verify it actually works
        torch.cuda.synchronize()
        _ = torch.tensor([1.0], device='cuda')
        torch.cuda.synchronize()
        _cuda_usable = True
    except (AssertionError, RuntimeError):
        _cuda_usable = False

    return _cuda_usable


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if is_cuda_usable():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def get_peak_gpu_memory_mb() -> float:
    """Get peak GPU memory usage in MB."""
    if is_cuda_usable():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def reset_gpu_memory_stats():
    """Reset GPU memory statistics."""
    if is_cuda_usable():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def get_cpu_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)
    except ImportError:
        return 0.0


class Timer:
    """Context manager for timing with GPU synchronization."""

    def __init__(self, use_cuda: bool = False):
        self.use_cuda = use_cuda and is_cuda_usable()
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        if self.use_cuda:
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.use_cuda:
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start_time


def run_scalability_test(
    n_spots: int,
    config: ExperimentConfig,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Run a single scalability test at a given data size.

    Parameters
    ----------
    n_spots : int
        Number of spots to generate
    config : ExperimentConfig
        Base experiment configuration
    seed : int
        Random seed
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        Dictionary containing timing, memory, and performance metrics
    """
    # Set all random seeds for reproducibility
    set_seed(seed)

    use_cuda = config.device == "cuda" and is_cuda_usable()

    # Reset memory tracking
    gc.collect()
    if use_cuda:
        reset_gpu_memory_stats()

    cpu_mem_start = get_cpu_memory_mb()

    # Scale anomaly counts proportionally
    base_n_spots = 3000
    scale_factor = n_spots / base_n_spots
    n_ectopic = max(10, int(config.n_ectopic * scale_factor))
    n_intrinsic = max(30, int(config.n_intrinsic * scale_factor))

    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing n_spots = {n_spots:,}")
        print(f"  Ectopic: {n_ectopic}, Intrinsic: {n_intrinsic}")
        print(f"  Device: {config.device}")
        print(f"{'='*60}")

    results = {
        "n_spots": n_spots,
        "n_genes": config.n_genes,
        "n_ectopic": n_ectopic,
        "n_intrinsic": n_intrinsic,
        "seed": seed,
        "device": config.device,
    }

    # 1. Data Generation Timing
    with Timer(use_cuda=False) as t_datagen:
        X, coords, labels, ectopic_idx, intrinsic_idx = generate_synthetic_data(
            n_spots=n_spots,
            n_genes=config.n_genes,
            n_ectopic=n_ectopic,
            n_intrinsic=n_intrinsic,
            n_modules=config.n_modules,
            min_distance_factor=config.min_distance_factor,
            random_state=seed,
        )
    results["time_datagen_sec"] = t_datagen.elapsed

    if verbose:
        print(f"  Data generation: {t_datagen.elapsed:.2f}s")
        print(f"  Actual ectopic: {len(ectopic_idx)}, intrinsic: {len(intrinsic_idx)}")

    # 2. Data Preparation (includes graph construction and tensor conversion)
    with Timer(use_cuda=use_cuda) as t_prep:
        data = prepare_data(
            X, coords,
            k=config.k_neighbors,
            device=config.device,
        )
    results["time_prep_sec"] = t_prep.elapsed

    gpu_mem_after_prep = get_gpu_memory_mb() if use_cuda else 0.0
    results["gpu_mem_after_prep_mb"] = gpu_mem_after_prep

    if verbose:
        print(f"  Data preparation: {t_prep.elapsed:.2f}s")
        if use_cuda:
            print(f"  GPU memory after prep: {gpu_mem_after_prep:.1f} MB")

    # 3. Model Creation
    with Timer(use_cuda=use_cuda) as t_model:
        model = InversePredictionModel(
            in_dim=data["n_genes"],
            hid_dim=config.hidden_dim,
            dropout=config.dropout,
        ).to(config.device)
    results["time_model_create_sec"] = t_model.elapsed

    # 4. Training
    reset_gpu_memory_stats() if use_cuda else None

    with Timer(use_cuda=use_cuda) as t_train:
        model = train_model(
            model,
            data["x_tensor"],
            data["coords_tensor"],
            data["edge_index"],
            n_epochs=config.n_epochs,
            lr=config.learning_rate,
            lambda_pos=config.lambda_pos,
            lambda_self=config.lambda_self,
            verbose=False,
        )
    results["time_train_sec"] = t_train.elapsed
    results["time_per_epoch_sec"] = t_train.elapsed / config.n_epochs

    gpu_mem_peak_train = get_peak_gpu_memory_mb() if use_cuda else 0.0
    results["gpu_mem_peak_train_mb"] = gpu_mem_peak_train

    if verbose:
        print(f"  Training ({config.n_epochs} epochs): {t_train.elapsed:.2f}s")
        print(f"  Time per epoch: {results['time_per_epoch_sec']*1000:.1f}ms")
        if use_cuda:
            print(f"  Peak GPU memory (train): {gpu_mem_peak_train:.1f} MB")

    # 5. Inference (score computation)
    reset_gpu_memory_stats() if use_cuda else None

    with Timer(use_cuda=use_cuda) as t_infer:
        scores_inv = compute_scores(
            model,
            data["x_tensor"],
            data["coords_tensor"],
            data["edge_index"],
            random_state=seed,
        )
    results["time_inference_sec"] = t_infer.elapsed

    gpu_mem_peak_infer = get_peak_gpu_memory_mb() if use_cuda else 0.0
    results["gpu_mem_peak_inference_mb"] = gpu_mem_peak_infer

    if verbose:
        print(f"  Inference: {t_infer.elapsed:.2f}s")
        if use_cuda:
            print(f"  Peak GPU memory (inference): {gpu_mem_peak_infer:.1f} MB")

    # 6. Total time for full pipeline
    results["time_total_sec"] = (
        results["time_datagen_sec"] +
        results["time_prep_sec"] +
        results["time_model_create_sec"] +
        results["time_train_sec"] +
        results["time_inference_sec"]
    )

    # CPU memory
    cpu_mem_end = get_cpu_memory_mb()
    results["cpu_mem_delta_mb"] = cpu_mem_end - cpu_mem_start
    results["cpu_mem_total_mb"] = cpu_mem_end

    # 7. Performance metrics (AUC)
    all_scores = {
        "Inv_PosError": scores_inv["s_pos"],
        "Inv_SelfRecon": scores_inv["s_self"],
        "Inv_NeighborRecon": scores_inv["s_neighbor"],
    }

    try:
        auc_df = compute_auc_metrics(all_scores, labels)

        for _, row in auc_df.iterrows():
            score_name = row["score"]
            results[f"auc_ectopic_{score_name}"] = row["auc_ectopic"]
            results[f"auc_intrinsic_{score_name}"] = row["auc_intrinsic"]

        # Primary metrics for summary
        pos_error_row = auc_df[auc_df["score"] == "Inv_PosError"].iloc[0]
        results["auc_ectopic"] = pos_error_row["auc_ectopic"]
        results["auc_intrinsic"] = pos_error_row["auc_intrinsic"]

        if verbose:
            print(f"  AUC (Ectopic, PosError): {results['auc_ectopic']:.3f}")
            print(f"  AUC (Intrinsic, PosError): {results['auc_intrinsic']:.3f}")

    except Exception as e:
        if verbose:
            print(f"  AUC computation failed: {e}")
        results["auc_ectopic"] = np.nan
        results["auc_intrinsic"] = np.nan

    # Cleanup
    del model, data, scores_inv, X, coords, labels
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    return results


def run(
    config: ExperimentConfig = None,
    n_spots_list: List[int] = None,
    n_seeds: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run scalability experiment across multiple data sizes.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration
    n_spots_list : list of int
        List of data sizes to test
    n_seeds : int
        Number of random seeds per data size
    verbose : bool
        Print progress

    Returns
    -------
    results : pd.DataFrame
        DataFrame with scalability results
    """
    if config is None:
        config = DEFAULT_CONFIG

    if n_spots_list is None:
        n_spots_list = [1000, 3000, 5000, 10000, 20000, 30000]

    seeds = list(range(42, 42 + n_seeds))

    all_results = []

    total_tests = len(n_spots_list) * len(seeds)
    test_num = 0

    if verbose:
        print("=" * 70)
        print("SCALABILITY EXPERIMENT FOR NATURE METHODS")
        print("=" * 70)
        print(f"Data sizes: {n_spots_list}")
        print(f"Seeds per size: {n_seeds}")
        print(f"Total tests: {total_tests}")
        print(f"Device: {config.device}")
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            try:
                print(f"GPU: {torch.cuda.get_device_name(0)}")
            except (AssertionError, RuntimeError) as e:
                print(f"GPU: detected but unavailable ({e})")
        print("=" * 70)

    for n_spots in n_spots_list:
        for seed in seeds:
            test_num += 1

            if verbose:
                print(f"\n[{test_num}/{total_tests}] n_spots={n_spots}, seed={seed}")

            try:
                result = run_scalability_test(
                    n_spots=n_spots,
                    config=config,
                    seed=seed,
                    verbose=verbose,
                )
                result["status"] = "success"
                all_results.append(result)

            except Exception as e:
                if verbose:
                    print(f"  FAILED: {e}")
                    traceback.print_exc()
                all_results.append({
                    "n_spots": n_spots,
                    "seed": seed,
                    "status": "failed",
                    "error": str(e),
                })

            # Force garbage collection between tests
            gc.collect()
            if is_cuda_usable():
                torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_results)

    # Compute summary statistics
    if verbose and len(results_df[results_df["status"] == "success"]) > 0:
        print("\n" + "=" * 70)
        print("SCALABILITY SUMMARY")
        print("=" * 70)

        success_df = results_df[results_df["status"] == "success"]

        summary = success_df.groupby("n_spots").agg({
            "time_total_sec": ["mean", "std"],
            "time_train_sec": ["mean", "std"],
            "time_inference_sec": ["mean", "std"],
            "gpu_mem_peak_train_mb": ["mean", "max"],
            "auc_ectopic": ["mean", "std"],
        }).round(3)

        print("\nTiming (mean +/- std):")
        for n_spots in n_spots_list:
            if n_spots in summary.index:
                row = summary.loc[n_spots]
                total_mean = row[("time_total_sec", "mean")]
                total_std = row[("time_total_sec", "std")]
                train_mean = row[("time_train_sec", "mean")]
                auc_mean = row[("auc_ectopic", "mean")]
                auc_std = row[("auc_ectopic", "std")]
                print(f"  n={n_spots:6,}: total={total_mean:6.1f}s +/- {total_std:5.1f}s, "
                      f"train={train_mean:5.1f}s, AUC={auc_mean:.3f}+/-{auc_std:.3f}")

    return results_df


def main():
    """Main entry point for scalability experiment."""
    import argparse

    parser = argparse.ArgumentParser(description="Scalability experiment for Nature Methods")
    parser.add_argument("--quick", action="store_true", help="Run quick test with fewer sizes/seeds")
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "exp13_scalability.csv"),
                        help="Output file path")
    parser.add_argument("--n_seeds", type=int, default=3, help="Number of seeds per data size")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--max_spots", type=int, default=30000, help="Maximum number of spots to test")
    args = parser.parse_args()

    # Configure experiment
    config = ExperimentConfig(
        n_epochs=args.n_epochs,
    )

    # Define data sizes to test
    if args.quick:
        n_spots_list = [1000, 3000, 5000]
        n_seeds = 2
    else:
        # Full experiment for Nature Methods
        n_spots_list = [1000, 3000, 5000, 10000, 20000, 30000]
        n_spots_list = [n for n in n_spots_list if n <= args.max_spots]
        n_seeds = args.n_seeds

    print(f"\nScalability Experiment Configuration:")
    print(f"  Data sizes: {n_spots_list}")
    print(f"  Seeds per size: {n_seeds}")
    print(f"  Epochs: {config.n_epochs}")
    print(f"  Device: {config.device}")

    # Run experiment
    results = run(
        config=config,
        n_spots_list=n_spots_list,
        n_seeds=n_seeds,
        verbose=True,
    )

    # Save results
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    # Print final summary table for paper
    success_df = results[results["status"] == "success"]
    if len(success_df) > 0:
        print("\n" + "=" * 70)
        print("TABLE FOR NATURE METHODS (Copy-paste ready)")
        print("=" * 70)

        summary = success_df.groupby("n_spots").agg({
            "time_total_sec": "mean",
            "time_train_sec": "mean",
            "time_inference_sec": "mean",
            "gpu_mem_peak_train_mb": "mean",
            "cpu_mem_total_mb": "mean",
            "auc_ectopic": "mean",
            "auc_intrinsic": "mean",
        }).round(2)

        summary.columns = [
            "Total (s)", "Train (s)", "Inference (s)",
            "GPU Mem (MB)", "CPU Mem (MB)",
            "AUC Ectopic", "AUC Intrinsic"
        ]

        print(summary.to_string())
        print("\n")

    return results


if __name__ == "__main__":
    main()
