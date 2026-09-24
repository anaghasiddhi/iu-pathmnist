
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load the unusual images selected by SQL.
unusual = pd.read_csv(
    PROJECT_ROOT / "output" / "unusual_images.csv"
)

# Load the original images.
with np.load(
    PROJECT_ROOT / "data" / "pathmnist.npz",
    allow_pickle=False
) as dataset:

    # Create a grid for displaying the selected images.
    fig, axes = plt.subplots(
        nrows=3,
        ncols=5,
        figsize=(15, 9)
    )

    axes = axes.flatten()

    for ax, row in zip(axes, unusual.itertuples()):
        # Example image_id: train_034415
        split, index = row.image_id.rsplit("_", 1)

        # Retrieve the image from its original split.
        image = dataset[f"{split}_images"][int(index)]

        ax.imshow(image)
        ax.set_title(
            f"{row.selection_reason}\n"
            f"{row.image_id} | Class {row.label}\n"
            f"Mean: {row.mean_intensity:.1f}, "
            f"SD: {row.std_intensity:.1f}",
            fontsize=8
        )
        ax.axis("off")

    # Hide unused panels.
    for ax in axes[len(unusual):]:
        ax.axis("off")

    plt.suptitle(
        "PathMNIST: Quantitatively Unusual Images",
        fontsize=15
    )

    plt.tight_layout()

    output_path = (
        PROJECT_ROOT / "output" / "unusual_images.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

print(f"Visualized {len(unusual)} unusual images")
print(f"Figure saved to: {output_path}")