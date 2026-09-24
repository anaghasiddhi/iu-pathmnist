
import numpy as np

# Open the downloaded dataset without changing it.
data = np.load("data/pathmnist.npz")

# Discover which arrays are stored in the file.
print("AVAILABLE ARRAYS")
print(data.files)

# Inspect the shape and data type of each array.
print("\nDATASET STRUCTURE")

for name in data.files:
    array = data[name]

    print(f"\n{name}")
    print(f"Shape: {array.shape}")
    print(f"Data type: {array.dtype}")

# Inspect the observed labels in each split.
print("\nLABEL DISTRIBUTIONS")

for name in data.files:
    if "labels" in name:
        labels, counts = np.unique(
            data[name],
            return_counts=True
        )

        print(f"\n{name}")

        for label, count in zip(labels, counts):
            print(f"Label {label}: {count}")

data.close()
