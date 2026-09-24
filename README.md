# PathMNIST: Medical Image Data Pipeline and Analysis

**Indiana University — Research Data Analyst Take-Home Assignment**


**Status:** Completed — Core implementation and reproducibility checks

## 1. Project Overview

This project develops a reproducible, CPU-based data engineering, analysis and machine-learning pipeline for the PathMNIST dataset from the MedMNIST project.

The pipeline ingests 107,180 medical images from an NPZ archive, validates dataset structure and labels, extracts image-level metadata, performs quality-control checks, detects exact duplicates using SHA-256 hashing, and stores structured results in DuckDB. SQL queries are used to analyze class distributions, compare image-intensity characteristics and identify unusual images for visualization.

A nine-class logistic regression classifier serves as a CPU-based baseline. Using the predefined dataset splits, model configurations were compared on validation data, and the selected model was evaluated on the held-out test set. Evaluation includes accuracy, balanced accuracy, per-class precision, recall, F1-score, specificity and a confusion matrix.

The project also examines data-leakage risks, scalability, incremental image processing and the limitations of aggregate evaluation metrics in medical imaging.

All five pipeline scripts have been successfully executed in sequence, and the resulting metadata, analyses, visualizations and model-evaluation outputs have been verified in the existing Python environment.

### Dataset clarification

The assignment specifies PathMNIST but also describes grayscale chest X-ray images with binary normal/pneumonia labels. The dataset choice was confirmed as PathMNIST by the interviewer.

Inspection of the downloaded file established that it contains 28 × 28 RGB images and nine numeric labels (0–8). Consequently, this implementation retains the original nine-class structure and adapts the relevant analyses to the multiclass dataset. It does not assign pneumonia labels to histopathology images.

## 2. Dataset

**Source:** [Official MedMNIST Zenodo distribution](https://zenodo.org/records/10519652)

**File:** `pathmnist.npz`

**Resolution:** 28 × 28 pixels

**Channels:** 3 (RGB)

**Observed labels:** 0–8

The NPZ archive contains six arrays: `train_images`, `train_labels`, `val_images`, `val_labels`, `test_images` and `test_labels`.

| Split      |      Images |
| ---------- | ----------: |
| Training   |      89,996 |
| Validation |      10,004 |
| Test       |       7,180 |
| **Total**  | **107,180** |

The dataset supplies predefined training, validation and test splits, which are preserved in the pipeline.


## 3. Environment and Setup

The project uses Python 3.12 and libraries suitable for a
CPU-based implementation. No GPU, CUDA, cloud infrastructure
or deep-learning framework is required.

### Dependencies

The primary dependencies are:

- Python 3.12
- NumPy
- Pandas
- DuckDB
- Matplotlib
- scikit-learn

Exact package versions, including supporting dependencies,
are recorded in `requirements.txt`.

### Installation

From the project root, create and activate a virtual
environment.

On Windows using Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

All pipeline scripts should be run from the project root
with the virtual environment activated.

### Download the Dataset

Download the standard 28 × 28 `pathmnist.npz` file from the
official MedMNIST Zenodo distribution:

https://zenodo.org/records/10519652

Create a `data` directory in the project root if one does
not already exist, and place the downloaded file at:

```text
data/pathmnist.npz
```

The expected project structure includes:

```text
iu-pathmnist/
├── data/
│   └── pathmnist.npz
├── src/
├── sql/
├── output/
├── README.md
└── requirements.txt
```

The dataset is not included in the GitHub repository.
It must be downloaded separately before running the pipeline.

The generated DuckDB database is also excluded from version
control and is recreated by running `src/build_dataset.py`.

## 4. Data Ingestion and Metadata

The ingestion pipeline is implemented in `src/build_dataset.py`.

It loads the NPZ archive, validates the three dataset splits and extracts one metadata record per image.

Each record contains:

| Field               | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| `image_id`          | Unique identifier combining split and image index            |
| `split`             | Train, validation or test                                    |
| `label`             | Original numeric class label                                 |
| `width`             | Image width in pixels                                        |
| `height`            | Image height in pixels                                       |
| `mean_intensity`    | Mean across all pixel/channel values                         |
| `std_intensity`     | Standard deviation across all pixel/channel values           |
| `min_intensity`     | Minimum pixel value                                          |
| `max_intensity`     | Maximum pixel value                                          |
| `image_hash`        | SHA-256 hash of the image's raw pixel data                   |
| `low_variance_flag` | Whether the image meets the low-variance screening criterion |

Image intensity statistics are calculated across all three RGB channels without modifying the original images.

The resulting metadata is stored in `output/pathmnist.duckdb`, in a table named `image_metadata`.

## 5. Data Validation and Quality Control

### Structural validation

The ingestion pipeline verifies image-label count agreement, image dimensions of 28 × 28 × 3, label-array shape and permitted numeric labels (0–8).

All three splits passed these initial checks.

### Database validation

An independent verification script, `src/verify_database.py`, checks the stored metadata.

| Check                              | Observed result |
| ---------------------------------- | --------------: |
| Total metadata records             |         107,180 |
| Unique image IDs                   |         107,180 |
| Missing metadata values            |               0 |
| Invalid stored statistics detected |               0 |
| Invalid stored dimensions detected |               0 |

These findings describe the implemented checks and do not rule out every possible source-data or scientific-quality issue.

### Exact duplicate detection

SHA-256 hashes were calculated from image pixel data and compared across the metadata records.

* Exact duplicate records detected: 0
* Duplicate hash groups: 0

Because the dataset uses a common image shape and dtype, matching pixel hashes provide a practical exact-duplicate check.

This does not exclude visually similar images, near-duplicates or multiple images originating from the same patient or source slide.

### Low-variance images

Images were screened using their pixel standard deviation. The lowest 1% of the observed distribution was used as the screening threshold.

**Observed threshold:** approximately 7.7746

| Split      | Flagged images |
| ---------- | -------------: |
| Training   |            957 |
| Validation |            114 |
| Test       |              1 |
| **Total**  |      **1,072** |

These images were flagged for investigation rather than automatically removed. Low pixel variance alone does not establish that an image is corrupted or unusable.

## 6. SQL Analysis

SQL queries are stored in the `sql/` directory and executed against the DuckDB metadata table using `src/analyze.py`.

### Class balance

Class counts and percentages were calculated for all nine labels in each dataset split.

The training and validation sets have closely matching class proportions. The test set has a noticeably different distribution.

For example, label 0 represents 10.41% of the training set and 18.64% of the test set.

This difference is relevant to the interpretation of model performance and aggregate evaluation metrics.

Detailed results are saved to `output/class_balance.csv`.

### Image characteristics

The average image intensity and average pixel standard deviation were calculated by class and dataset split.

Selected training-set observations:

| Label | Average mean intensity | Average pixel SD |
| ----- | ---------------------: | ---------------: |
| 0     |                207.608 |           22.936 |
| 1     |                142.268 |           23.878 |
| 2     |                165.076 |           36.514 |
| 6     |                165.693 |           38.272 |
| 8     |                157.743 |           38.197 |

In the training set, label 0 has the highest average mean intensity and label 1 has the lowest.

The class-1 average mean intensity also differs between training (142.268) and testing (112.177). This is an observed distributional difference; its cause has not been established.

The complete results are saved to `output/image_characteristics.csv`.

### Unusual-image analysis

Images were ranked using three quantitative criteria:

1. Lowest mean pixel intensity.
2. Highest mean pixel intensity.
3. Lowest pixel standard deviation.

The five highest-ranked images under each criterion were selected. Overlapping selections were deduplicated and assigned one primary selection reason.

This produced 13 unique images:

| Primary selection reason | Images |
| ------------------------ | -----: |
| Unusually bright         |      5 |
| Unusually dark           |      5 |
| Low contrast             |      3 |

All five brightest images belong to label 2 and have mean intensities around 244.

All five darkest images belong to label 1, with means ranging from approximately 7 to 45.

The three remaining lowest-contrast images belong to label 5, with pixel standard deviations of approximately 1.7–2.2.

The bright and low-contrast groups overlap because some almost-uniform images are also extremely bright.

The selected examples were retrieved from the original NPZ file and displayed in `output/unusual_images.png`.

These are descriptive, quantitative findings. No claim is made that the selected images are medically abnormal, corrupted or incorrectly labeled.


## 7. Baseline Classifier

### Model and preprocessing

A nine-class logistic regression classifier was implemented using
scikit-learn. Each 28 × 28 RGB image was flattened into 2,352
numerical features and normalized by dividing pixel values by 255.

A reproducible, stratified subset of 20,000 images was selected
from the predefined training split using `random_state=42`.
The original validation and test splits were preserved.

Logistic regression was selected as a straightforward CPU-based
baseline. It does not directly model the complex spatial
relationships present in histopathology images.

### Model selection

Three configurations were compared using the validation set:

| Configuration | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| C=1.0, 100 iterations | 38.01% | 37.80% | 0.363 |
| C=1.0, 300 iterations | 37.83% | 37.49% | 0.371 |
| C=0.1, 300 iterations | 40.08% | 39.38% | 0.387 |

The final configuration used `C=0.1`, `max_iter=300`,
the `lbfgs` solver and `random_state=42`.

The optimizer reached its maximum iteration count without
satisfying its convergence criterion. This limitation is
retained in the interpretation of the model results.

### Final test evaluation

The selected model was evaluated on the previously untouched
test split containing 7,180 images.

| Metric | Test result |
|---|---:|
| Accuracy | 47.60% |
| Balanced accuracy | 40.50% |
| Macro F1 | 0.384 |
| Weighted F1 | 0.461 |

Class-specific performance varied considerably.

Class 0 achieved approximately 81.2% recall and 90.7%
precision. Class 1 achieved 100% recall but only 57.4%
precision, indicating that other classes were sometimes
incorrectly predicted as class 1.

Class 7 achieved approximately 9.7% recall, demonstrating
that the model struggled to recognize this category.

The difference between overall and balanced accuracy
reinforces the importance of examining per-class performance,
especially when class distributions differ between splits.

### Evaluation outputs

The following files were generated:

- `output/test_metrics.csv`: per-class precision, recall,
  F1, support and one-vs-rest specificity.
- `output/model_evaluation.png`: nine-class test
  confusion matrix.

No additional model selection was performed using the test
results.


## 8. Research and Data Engineering Considerations

### 8.1 Data leakage

The principal leakage risks considered were patient- or
slide-level overlap, exact and near-duplicate images across
dataset splits, preprocessing leakage and test-set
contamination during model selection.

The pipeline generated SHA-256 hashes from image pixel data
and detected no exact duplicate images. However, exact
hashing cannot identify every near-duplicate or overlapping
image patch.

The available NPZ arrays do not contain patient or
source-slide identifiers. Consequently, patient-level and
slide-level separation cannot be independently verified
using the supplied data alone.

Preprocessing divided pixel values by a fixed constant of
255, without fitting normalization parameters using
validation or test data.

Model configurations were compared using the predefined
validation split. The test split was reserved for final
evaluation.

These safeguards address several leakage risks, but the
available provenance information is insufficient to rule
out all possible medical-imaging leakage.

### 8.2 Scaling to one million images

If the pipeline needed to process one million medical images
arriving continuously from several hospitals, I would
prioritize three architectural changes.

**1. Parallel and memory-efficient preprocessing**

I would process images in bounded batches and parallelize
independent operations, such as image validation, metadata
extraction and hashing, across multiple CPU workers.

Worker counts would be determined through benchmarking,
since disk I/O or memory bandwidth could become the
bottleneck before CPU processing.

**2. Incremental processing**

I would replace full dataset reprocessing with incremental
ingestion supported by a persistent manifest.

Successfully processed images would retain their existing
metadata. New or changed images would be processed
separately, with failed records eligible for controlled
retries.

**3. Source-level validation and provenance**

I would preserve hospital, patient and source-slide
identifiers when available, alongside imaging metadata
and processing status.

Source-specific validation would help identify differences
in image dimensions, formats, staining, scanners and
other acquisition characteristics.

These measures would improve scalability while supporting
data quality, traceability and leakage prevention.

### 8.3 Incremental processing

To process 10,000 newly arriving images without
recalculating metadata for the existing one million,
I would extend the current SHA-256 hashing approach with
a persistent ingestion manifest.

The manifest would record source identifiers, file
locations, content hashes, processing statuses, timestamps
and failure reasons.

Incoming records would be handled according to their
previously recorded state:

- New images would undergo validation and metadata
  extraction before being appended to the database.
- Exact duplicates would be identified through matching
  hashes and would not require redundant metadata
  computation.
- Previously failed images would be identified through
  their stored processing status and retried when
  appropriate.

For images that cannot be decoded, the hash of the original
file bytes could still be retained alongside the source
identifier and failure record.

Database uniqueness constraints, transactions and
idempotent writes would help prevent duplicate records
after interruptions or repeated ingestion attempts.

Existing successfully processed records would remain
unchanged unless their source content or processing
requirements changed.

### 8.4 Scientifically misleading results

A data pipeline can execute successfully and produce
technically valid outputs while still supporting misleading
scientific conclusions.

One example is evaluating a multiclass classifier using
aggregate metrics without examining performance for each
individual class.

In this project, the logistic regression baseline achieved
47.60% test accuracy but only 40.50% balanced accuracy.
Its class-specific recall ranged from 100% for class 1
to approximately 9.7% for class 7.

Reporting only overall accuracy would conceal these
differences and could overstate the model's reliability
for particular tissue categories.

Similarly, AUROC alone would not establish satisfactory
precision or recall at a specific operational threshold,
particularly in imbalanced datasets. AUROC was not
calculated in this baseline experiment.

To reduce the risk of misleading interpretation, I
reported balanced accuracy, per-class precision, recall,
F1 and specificity, and examined the complete confusion
matrix.

I also compared class distributions across the predefined
splits and documented the limitations of the available
provenance information.

These practices distinguish successful code execution
from scientifically reliable evaluation.


## 9. Running the Project

After installing dependencies and downloading the PathMNIST dataset, run the following commands from the project root.

### 9.1 Build and Validate the Dataset

```bash
python src/build_dataset.py
python src/verify_database.py
```

The first script loads the PathMNIST dataset, validates its structure, extracts image-level metadata, performs quality-control checks, detects exact duplicates using SHA-256 hashes, flags low-variance images, and stores the results in DuckDB.

The second script independently verifies the stored metadata, including record counts, dataset splits, unique image identifiers, invalid statistics, and image dimensions.

### 9.2 Run SQL Analysis

```bash
python src/analyze.py
python src/plot_unusual.py
```

The analysis script executes SQL queries against the DuckDB database to investigate class balance, image characteristics, and quantitatively unusual images.

The visualization script retrieves the selected unusual images from the original NPZ file and generates a figure showing unusually bright, dark, and low-contrast examples.

### 9.3 Train and Evaluate the Baseline Classifier

```bash
python src/model.py
```

This script loads the predefined dataset splits, flattens and normalizes the RGB images, and trains a nine-class logistic regression classifier using a reproducible, stratified subset of 20,000 training images.

The selected configuration uses C=0.1 and a maximum of 300 iterations. The script reports validation performance and evaluates the selected classifier on the held-out test set.

It calculates accuracy, balanced accuracy, per-class precision, recall, F1-score, one-vs-rest specificity, and the confusion matrix.

Running the script retrains the classifier and regenerates its evaluation outputs. The optimizer's convergence warning is a documented limitation.

### 9.4 Generated Outputs

The pipeline generates the following files:

| File | Description |
|---|---|
| `output/pathmnist.duckdb` | Queryable image-level metadata |
| `output/class_balance.csv` | Class counts and percentages by split |
| `output/image_characteristics.csv` | Intensity statistics by class and split |
| `output/unusual_images.csv` | Quantitatively selected unusual images |
| `output/unusual_images.png` | Visualization of unusual images |
| `output/test_metrics.csv` | Per-class test evaluation metrics |
| `output/model_evaluation.png` | Nine-class test confusion matrix |

The original PathMNIST dataset and generated DuckDB database are excluded from version control and can be recreated using the provided instructions.


## 10. Limitations

### Data Quality and Provenance

**Exact versus near-duplicate images**

SHA-256 hashing identified no exact duplicate pixel arrays
among the 107,180 images. However, exact hashing cannot
detect images that differ slightly due to cropping, resizing,
rotation or other transformations.

The absence of exact duplicates also does not establish
patient-level or tissue-slide-level independence.

**Incomplete provenance**

The supplied NPZ archive does not contain patient or
source-slide identifiers. Consequently, the implemented
pipeline cannot independently verify complete separation
at these levels.

Although the original MedMNIST project provides additional
source-image mapping information, this was not incorporated
into the current analysis.

**Limited image-quality assessment**

Mean intensity, pixel standard deviation, minimum intensity
and maximum intensity provide useful quantitative image
characteristics, but they do not fully describe histological
image quality.

The implemented pipeline does not comprehensively assess
staining consistency, tissue morphology, blur, acquisition
artifacts or biological suitability.

**Low-variance screening**

The lowest 1% of images by pixel standard deviation were
flagged for review, resulting in 1,072 flagged images.

This percentile-based threshold is a statistical screening
criterion, not a clinical or biological validity threshold.
Flagged images were retained rather than automatically
excluded.

### Dataset and Evaluation

**Differences between dataset splits**

The training and validation sets have closely matching
class distributions, whereas the test set has different
class proportions and image-intensity characteristics.

According to the official PathMNIST documentation, the
training and validation images originate from
NCT-CRC-HE-100K, while the test images originate from
CRC-VAL-HE-7K, a different clinical center.

The extent to which acquisition, staining, source-population
or other factors explain the observed differences was not
established in this analysis.

**Limited feature representation**

The baseline classifier uses flattened RGB pixel values.
This simple representation does not explicitly model spatial
relationships, tissue morphology or complex histological
patterns.

A more sophisticated image-classification approach may
capture additional information, but developing one was
outside the scope of this CPU-based baseline.

**Incomplete optimizer convergence**

The selected logistic regression configuration used
C=0.1 and a maximum of 300 iterations.

The LBFGS optimizer reached the iteration limit without
satisfying its convergence criterion. The reported results
are therefore based on a model whose numerical optimization
was not fully completed.

**Uneven class-specific performance**

The baseline achieved 47.60% overall test accuracy and
40.50% balanced accuracy.

Performance varied substantially between tissue classes.
In particular, class 1 achieved 100% recall but only
approximately 57.4% precision, while class 7 achieved
approximately 9.7% recall.

Overall accuracy alone would conceal these differences.
The model's performance is insufficient to establish
reliability for clinical use.

**Limited model optimization**

Only three logistic regression configurations were compared,
using a reproducible stratified subset of 20,000 training
images.

No extensive hyperparameter search, alternative model
benchmarking or external clinical validation was performed.
The test set was reserved for the final evaluation.


## 11. AI Assistance

ChatGPT was used to interpret assignment requirements,
plan the project structure, explain Python and SQL
concepts, draft portions of the implementation and
documentation, assist with debugging, and support
interpretation of analytical and model-evaluation results.

Generated code was reviewed and tested against the
downloaded dataset. Dataset structure, class labels,
metadata counts, database contents, SQL outputs and
model predictions were checked through executed code.

An important AI-assisted correction concerned the
assignment's dataset description. The written instructions
specified PathMNIST but described a binary
normal/pneumonia classification task.

Rather than accepting that interpretation, I inspected
the actual NPZ file and confirmed that it contains
28 × 28 RGB images with nine numeric classes.
After the dataset choice was confirmed by the interviewer,
the analysis and baseline classifier were adapted to
retain the original nine-class structure.

AI assistance was also used while exploring logistic
regression configurations and implementing the
multiclass evaluation. The results were checked against
validation and test outputs, and the unresolved optimizer
convergence warning was retained as a documented
limitation.

## 12. Current Status

The following components have been implemented:

- Dataset ingestion and structural validation
- Image-level metadata extraction
- SHA-256 exact-duplicate detection
- Low-variance image screening
- DuckDB storage and independent database verification
- SQL class-balance analysis
- SQL image-characteristic analysis
- Unusual-image selection and visualization
- Nine-class CPU-based logistic regression baseline
- Validation-based model selection
- Held-out test evaluation
- Per-class precision, recall, F1 and specificity
- Test confusion-matrix visualization
- Research and data-engineering judgment responses


The core implementation, documentation, dependency recording,
and GitHub upload are complete.

All five pipeline scripts were successfully executed in
sequence in the existing Python virtual environment. The
reproducibility run generated the expected metadata, SQL
analysis results, visualizations, and model evaluation outputs.

The project has not been independently tested in a newly
created virtual environment.