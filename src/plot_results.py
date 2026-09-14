from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

CSV = ROOT / "results" / "summary" / "benchmark_results.csv"
PLOTS = ROOT / "results" / "plots"

PLOTS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)


def plot_configuration(image_size, batch_size, filename):
    data = df[
        (df["image_size"] == image_size)
        & (df["batch_size"] == batch_size)
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        data["workers"],
        data["pytorch"],
        marker="o",
        label="PyTorch DataLoader",
    )

    plt.plot(
        data["workers"],
        data["minimal"],
        marker="o",
        label="Minimal Dataset DataLoader",
    )

    plt.xlabel("Number of workers")
    plt.ylabel("Throughput (samples/s)")
    plt.title(
        f"Food-101 — {image_size}×{image_size}, "
        f"batch size {batch_size}"
    )

    plt.xticks(data["workers"])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output = PLOTS / filename
    plt.savefig(output, dpi=150)
    plt.close()

    print(f"Created: {output}")


plot_configuration(
    224,
    64,
    "food101_224_bs64.png",
)

plot_configuration(
    64,
    64,
    "food101_64_bs64.png",
)

plot_configuration(
    64,
    256,
    "food101_64_bs256.png",
)
