from pathlib import Path
import time
import statistics

import torch
from torch.utils.data import DataLoader as TorchDataLoader

from minimal_dataset import (
    ParquetDataset,
    DataLoader as MinimalDataLoader,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PARQUET_FILE = (
    PROJECT_ROOT
    / "data"
    / "parquet"
    / "food101_train_10k.parquet"
)

NUM_SAMPLES = 10_000
BATCH_SIZE = 64
WORKERS = [1, 2, 4]
REPEATS = 3
IMAGE_SIZE = (224, 224)


def collate_batch(samples):
    images = torch.stack([
        sample[0]
        for sample in samples
    ])

    if images.dtype == torch.uint8:
        images = images.float()
        images.div_(255.0)

    labels = torch.tensor([
        sample[1]
        for sample in samples
    ])

    return images, labels


def measure_loader(loader):
    start = time.perf_counter()

    total_samples = 0
    total_batches = 0

    for images, labels in loader:
        total_samples += len(labels)
        total_batches += 1

    elapsed = time.perf_counter() - start

    throughput = total_samples / elapsed

    return {
        "samples": total_samples,
        "batches": total_batches,
        "time": elapsed,
        "throughput": throughput,
    }


def benchmark_custom(dataset, workers):
    results = []

    for repeat in range(REPEATS):
        loader = MinimalDataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=workers,
            collate_fn=collate_batch,
        )

        result = measure_loader(loader)
        results.append(result)

        print(
            f"  repeat {repeat + 1}: "
            f"{result['throughput']:.1f} samples/s "
            f"({result['time']:.2f}s)"
        )

    return results


def benchmark_pytorch(dataset, workers):
    results = []

    for repeat in range(REPEATS):
        loader = TorchDataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=workers,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_batch,
            persistent_workers=False,
        )

        result = measure_loader(loader)
        results.append(result)

        print(
            f"  repeat {repeat + 1}: "
            f"{result['throughput']:.1f} samples/s "
            f"({result['time']:.2f}s)"
        )

    return results


def summarize(results):
    throughputs = [
        result["throughput"]
        for result in results
    ]

    times = [
        result["time"]
        for result in results
    ]

    return {
        "throughput_mean": statistics.mean(throughputs),
        "throughput_std": statistics.stdev(throughputs),
        "time_mean": statistics.mean(times),
        "time_std": statistics.stdev(times),
    }


def main():
    print("Loading Parquet dataset...")

    dataset = ParquetDataset(
        parquet_path=str(PARQUET_FILE),
        max_samples=NUM_SAMPLES,
        image_size=IMAGE_SIZE,
    )

    print(f"Dataset: {len(dataset):,} samples")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMAGE_SIZE}")
    print()

    final_results = []

    for workers in WORKERS:
        print("=" * 60)
        print(f"Workers: {workers}")
        print("=" * 60)

        print()
        print("Minimal Dataset DataLoader")

        custom_results = benchmark_custom(
            dataset,
            workers,
        )

        custom_summary = summarize(
            custom_results
        )

        print()
        print("PyTorch DataLoader")

        torch_results = benchmark_pytorch(
            dataset,
            workers,
        )

        torch_summary = summarize(
            torch_results
        )

        final_results.append({
            "workers": workers,
            "custom": custom_summary,
            "torch": torch_summary,
        })

        print()

    print()
    print("=" * 72)
    print("FINAL RESULTS")
    print("=" * 72)

    print(
        f"{'Workers':<10}"
        f"{'PyTorch':>20}"
        f"{'Minimal Dataset':>20}"
        f"{'Speedup':>15}"
    )

    for result in final_results:
        workers = result["workers"]

        torch_tp = (
            result["torch"]["throughput_mean"]
        )

        custom_tp = (
            result["custom"]["throughput_mean"]
        )

        speedup = custom_tp / torch_tp

        print(
            f"{workers:<10}"
            f"{torch_tp:>17.1f} s/s"
            f"{custom_tp:>17.1f} s/s"
            f"{speedup:>14.2f}x"
        )


if __name__ == "__main__":
    main()
