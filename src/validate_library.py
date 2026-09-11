from pathlib import Path

import torch
import minimal_dataset
from minimal_dataset import ParquetDataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PARQUET_FILE = (
    PROJECT_ROOT
    / "data"
    / "parquet"
    / "food101_train_10k.parquet"
)


def main():
    print("minimal-dataset version:", minimal_dataset.__version__)
    print("PyTorch version:", torch.__version__)
    print()

    print("Loading ParquetDataset...")

    dataset = ParquetDataset(
        parquet_path=str(PARQUET_FILE),
        max_samples=512,
        image_size=(224, 224),
    )

    print(f"Dataset size: {len(dataset):,}")

    image, label = dataset[0]

    print()
    print("Single sample")
    print("-------------")
    print("Shape:", image.shape)
    print("Dtype:", image.dtype)
    print("Label:", label)

    assert image.shape == (3, 224, 224)
    assert image.dtype == torch.uint8

    print()
    print("Creating custom DataLoader...")

    loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
    )

    total_samples = 0

    for batch_index, (images, labels) in enumerate(loader):
        if batch_index == 0:
            print()
            print("First batch")
            print("-----------")
            print("Images:", images.shape)
            print("Images dtype:", images.dtype)
            print("Labels:", labels.shape)
            print("Labels dtype:", labels.dtype)

        total_samples += len(labels)

    print()
    print(f"Samples processed: {total_samples:,}")
    print("Custom DataLoader validation successful.")


if __name__ == "__main__":
    main()
