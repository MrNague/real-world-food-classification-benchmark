import argparse
import time

import torch
from minimal_dataset import ParquetDataset


DATASETS = {
    "tiny": "/fscratch/nague/storage_benchmarks/images.parquet",
    "food": "data/parquet/food101_train_10k.parquet",
}

NUM_SAMPLES = 5_000
IMAGE_SIZE = (64, 64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["tiny", "food"],
        required=True,
    )
    args = parser.parse_args()

    torch.set_num_threads(1)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    path = DATASETS[args.dataset]

    print(f"Dataset: {args.dataset}")
    print(f"Path: {path}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Samples measured: {NUM_SAMPLES}")
    print()

    print("Loading ParquetDataset...", flush=True)
    load_start = time.perf_counter()

    dataset = ParquetDataset(
        parquet_path=path,
        max_samples=NUM_SAMPLES,
        image_size=IMAGE_SIZE,
    )

    load_elapsed = time.perf_counter() - load_start

    print(f"Dataset loaded in {load_elapsed:.3f} s")
    print(f"Dataset length exposed: {len(dataset)}")
    print()

    print("Warm-up...", flush=True)

    for i in range(100):
        _ = dataset[i]

    print("Benchmarking __getitem__...", flush=True)

    start = time.perf_counter()

    for i in range(NUM_SAMPLES):
        _ = dataset[i]

    elapsed = time.perf_counter() - start
    throughput = NUM_SAMPLES / elapsed
    mean_ms = elapsed / NUM_SAMPLES * 1000

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Load time: {load_elapsed:.3f} s")
    print(f"Processing time: {elapsed:.3f} s")
    print(f"Throughput: {throughput:.1f} samples/s")
    print(f"Mean sample time: {mean_ms:.3f} ms")


if __name__ == "__main__":
    main()
