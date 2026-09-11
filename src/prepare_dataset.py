from pathlib import Path
import random

import pyarrow as pa
import pyarrow.parquet as pq
from torchvision.datasets import Food101


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"

OUTPUT_FILE = PARQUET_DIR / "food101_train_10k.parquet"

NUM_SAMPLES = 10_000
SEED = 42


def download_food101():
    print("Downloading / loading Food-101...")

    dataset = Food101(
        root=RAW_DIR,
        split="train",
        download=True,
    )

    print(f"Training samples available: {len(dataset):,}")
    print(f"Number of classes: {len(dataset.classes)}")

    return dataset


def select_samples(dataset):
    rng = random.Random(SEED)

    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    selected = indices[:NUM_SAMPLES]

    return selected


def create_parquet(dataset, indices):
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    images = []
    labels = []

    print(f"Creating Parquet dataset with {len(indices):,} samples...")

    for position, index in enumerate(indices, start=1):
        image_path = Path(dataset._image_files[index])
        label = dataset._labels[index]

        # Keep the original JPEG bytes.
        image_bytes = image_path.read_bytes()

        images.append(image_bytes)
        labels.append(label)

        if position % 1000 == 0:
            print(f"Processed {position:,}/{len(indices):,} images")

    table = pa.table(
        {
            "image": pa.array(images, type=pa.binary()),
            "label": pa.array(labels, type=pa.int64()),
        }
    )

    pq.write_table(
        table,
        OUTPUT_FILE,
        compression="snappy",
        row_group_size=1024,
    )

    print()
    print("Parquet creation completed.")
    print(f"File: {OUTPUT_FILE}")
    print(f"Rows: {table.num_rows:,}")
    print(f"Size: {OUTPUT_FILE.stat().st_size / (1024 ** 2):.2f} MB")


def verify_parquet():
    table = pq.read_table(OUTPUT_FILE)

    print()
    print("Verification")
    print("------------")
    print(table.schema)
    print(f"Rows: {table.num_rows:,}")

    assert table.num_rows == NUM_SAMPLES
    assert "image" in table.column_names
    assert "label" in table.column_names

    print("Parquet dataset is valid.")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    dataset = download_food101()
    indices = select_samples(dataset)

    create_parquet(dataset, indices)
    verify_parquet()


if __name__ == "__main__":
    main()
