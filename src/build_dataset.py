
from pathlib import Path
import numpy as np
import hashlib

import pandas as pd
import duckdb

# Find the project directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Locate the dataset.
DATA_PATH = PROJECT_ROOT / "data" / "pathmnist.npz"

# Open the NPZ archive.
dataset = np.load(DATA_PATH, allow_pickle=False)

print("Available arrays:", dataset.files)


# Validate the training, validation, and test splits.
splits = ["train", "val", "test"]

for split in splits:
    images = dataset[f"{split}_images"]
    labels = dataset[f"{split}_labels"]

    print(f"\nChecking {split} split")

    # Check that every image has a corresponding label.
    assert len(images) == len(labels), (
        f"{split}: image and label counts do not match"
    )

    # Check image dimensions and channels.
    assert images.ndim == 4
    assert images.shape[1:] == (28, 28, 3)

    # Verify that labels have the expected shape.
    assert labels.shape == (len(images), 1)

    # Verify the observed label values.
    assert np.isin(labels, np.arange(9)).all()

    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")
    print(f"Image shape: {images.shape[1:]}")
    print("Validation passed")


# Build one metadata record per image.
metadata = []

for split in splits:
    images = dataset[f"{split}_images"]
    labels = dataset[f"{split}_labels"].flatten()

    print(f"Processing {split} metadata...")

    for index, image in enumerate(images):
        record = {
            "image_id": f"{split}_{index:06d}",
            "split": split,
            "label": int(labels[index]),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "mean_intensity": float(image.mean()),
            "std_intensity": float(image.std()),
            "min_intensity": int(image.min()),
            "max_intensity": int(image.max()),
            "image_hash": hashlib.sha256(image.tobytes()).hexdigest(),
        }

        metadata.append(record)

print(f"\nTotal metadata records: {len(metadata)}")
print("\nFirst record:")
print(metadata[0])


df = pd.DataFrame(metadata)

print("\nMetadata table shape:", df.shape)
print(df.head())

duplicate_mask = df.duplicated(
    subset=["image_hash"],
    keep=False
)

duplicates = df[duplicate_mask]

print("\nDuplicate image records:", len(duplicates))
print(
    "Duplicate hash groups:",
    duplicates["image_hash"].nunique()
)

print("\nMissing values:")
print(df.isna().sum())

threshold = df["std_intensity"].quantile(0.01)

df["low_variance_flag"] = (
    df["std_intensity"] <= threshold
)

print("\nLow-variance threshold:", threshold)
print(
    "Flagged images:",
    df["low_variance_flag"].sum()
)
import duckdb

output_dir = PROJECT_ROOT / "output"
output_dir.mkdir(exist_ok=True)

database_path = output_dir / "pathmnist.duckdb"

with duckdb.connect(str(database_path)) as con:
    con.register("metadata_df", df)
    con.execute("""
        CREATE OR REPLACE TABLE image_metadata AS
        SELECT * FROM metadata_df
    """)

    count = con.execute(
        "SELECT COUNT(*) FROM image_metadata"
    ).fetchone()[0]

    print("\nRows stored in DuckDB:", count)