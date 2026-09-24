
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "pathmnist.npz"

# Load the predefined dataset splits.
with np.load(DATA_PATH, allow_pickle=False) as data:
    X_train = data["train_images"]
    y_train = data["train_labels"].ravel()

    X_val = data["val_images"]
    y_val = data["val_labels"].ravel()

    X_test = data["test_images"]
    y_test = data["test_labels"].ravel()

print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


# Flatten each RGB image and normalize pixel values.
def prepare_images(images):
    return images.reshape(len(images), -1).astype(np.float32) / 255.0


X_train = prepare_images(X_train)
X_val = prepare_images(X_val)
X_test = prepare_images(X_test)

print("\nPrepared image shapes:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

print("\nTraining pixel range:")
print("Minimum:", X_train.min())
print("Maximum:", X_train.max())


# Select a reproducible, class-stratified training subset.
X_subset, _, y_subset, _ = train_test_split(
    X_train,
    y_train,
    train_size=20000,
    stratify=y_train,
    random_state=42
)

print("\nTraining subset:", X_subset.shape)
print("Class distribution:", np.bincount(y_subset))

# Train a simple CPU-based baseline.

model = LogisticRegression(
    solver="lbfgs",
    max_iter=300,
    C=0.1,
    random_state=42
)

print("\nTraining logistic regression...")
start = time.perf_counter()

model.fit(X_subset, y_subset)

elapsed = time.perf_counter() - start
print(f"Training completed in {elapsed:.1f} seconds")
print("Iterations:", model.n_iter_)


# Evaluate on the predefined validation split.
val_predictions = model.predict(X_val)

print("\nVALIDATION RESULTS")
print(
    "Accuracy:",
    accuracy_score(y_val, val_predictions)
)
print(
    "Balanced accuracy:",
    balanced_accuracy_score(y_val, val_predictions)
)

print("\nClassification report:")
print(
    classification_report(
        y_val,
        val_predictions,
        labels=np.arange(9),
        digits=3,
        zero_division=0
    )
)

print("\nConfusion matrix:")
print(
    confusion_matrix(
        y_val,
        val_predictions,
        labels=np.arange(9)
    )
)

# Final evaluation on the untouched test split.
test_predictions = model.predict(X_test)

print("\nFINAL TEST RESULTS")
print("Test accuracy:", accuracy_score(y_test, test_predictions))
print(
    "Test balanced accuracy:",
    balanced_accuracy_score(y_test, test_predictions)
)

print("\nTest classification report:")
print(
    classification_report(
        y_test,
        test_predictions,
        labels=np.arange(9),
        digits=3,
        zero_division=0
    )
)

test_cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=np.arange(9)
)

print("\nTest confusion matrix:")
print(test_cm)

# Create output directory for saved results.
output_dir = PROJECT_ROOT / "output"
output_dir.mkdir(exist_ok=True)

# Build a structured classification report.
report_dict = classification_report(
    y_test,
    test_predictions,
    labels=np.arange(9),
    digits=3,
    zero_division=0,
    output_dict=True
)

# Compute one-vs-rest specificity for each class.
specificity_rows = []

for class_idx in range(9):
    tp = test_cm[class_idx, class_idx]
    fn = test_cm[class_idx, :].sum() - tp
    fp = test_cm[:, class_idx].sum() - tp
    tn = test_cm.sum() - (tp + fn + fp)

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    specificity_rows.append(
        {
            "class": class_idx,
            "precision": report_dict[str(class_idx)]["precision"],
            "recall": report_dict[str(class_idx)]["recall"],
            "f1_score": report_dict[str(class_idx)]["f1-score"],
            "support": report_dict[str(class_idx)]["support"],
            "specificity": specificity,
        }
    )

metrics_df = pd.DataFrame(specificity_rows)

print("\nPER-CLASS TEST METRICS")
print(metrics_df.to_string(index=False))

metrics_path = output_dir / "test_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)

print(f"\nPer-class metrics saved to: {metrics_path}")

# Plot confusion matrix.
plt.figure(figsize=(10, 8))
plt.imshow(test_cm, interpolation="nearest", cmap="Blues")
plt.title("PathMNIST Test Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(9)
plt.xticks(tick_marks, tick_marks)
plt.yticks(tick_marks, tick_marks)
plt.xlabel("Predicted label")
plt.ylabel("True label")

# Annotate each cell with the count.
threshold = test_cm.max() / 2.0

for i in range(test_cm.shape[0]):
    for j in range(test_cm.shape[1]):
        plt.text(
            j,
            i,
            str(test_cm[i, j]),
            ha="center",
            va="center",
            color="white" if test_cm[i, j] > threshold else "black",
            fontsize=8
        )

plt.tight_layout()

figure_path = output_dir / "model_evaluation.png"
plt.savefig(figure_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Confusion matrix figure saved to: {figure_path}")