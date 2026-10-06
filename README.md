# Auditing Before Automation: Shortcut-Aware CXR Triage

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](Auditing_Before_Automation_CXR_Triage_R1.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hamamh66/Auditing-Before-Automation-CXR-Triage/blob/main/Auditing_Before_Automation_CXR_Triage_R1.ipynb)

This repository contains the complete computational workflow supporting the manuscript:

> **Auditing Before Automation: Shortcut-Aware Hierarchical Selective Triage of Pneumonia and Tuberculosis on Chest Radiographs**  
> Fatma Mallek, Rahma Zayoud, and Habib Hamam

The project studies three-class chest-radiograph classification—normal, pneumonia, and tuberculosis—while treating data integrity and shortcut detection as prerequisites for any automation claim. Its central lesson is deliberately cautious: strong internal discrimination is not evidence of clinical generalization when simple acquisition metadata can reproduce nearly the same result.

## Why this repository matters

Public medical-image compilations can contain duplicated images, contradictory labels, and class-linked acquisition signatures. A model can therefore appear highly accurate while learning dataset provenance rather than pathology. This workflow places auditing before model interpretation and lets a metadata-only negative control constrain the scientific conclusion.

Two notebooks are provided. `Auditing_Before_Automation_CXR_Triage_R1.ipynb` (notebook version R1-1.1) is the current workflow and the one whose outputs are reported in the revised manuscript; it extends the original submission with forensic, ablation, robustness, and second-dataset analyses and uses deterministic full-precision feature extraction. `Auditing_Before_Automation_CXR_Triage.ipynb` is the original submitted workflow, kept for provenance.

The R1 notebook integrates:

- SHA-256 exact-identity auditing;
- 64-bit perceptual hashing with BK-tree near-duplicate search;
- conservative removal of contradictory-label identities;
- duplicate-group-aware train/validation/calibration/test partitioning;
- one-pass frozen MobileNetV3-Small feature extraction;
- direct multiclass and hierarchical classification;
- validation-only temperature scaling;
- paired stratified bootstrap confidence intervals;
- class-conditional conformal prediction;
- calibration-locked selective referral;
- a seven-variable metadata-only shortcut control with dimensions-only and intensity-only ablations;
- a forensic characterization of the excluded files, including the class–resolution association before and after exclusion;
- class-wise acceptance, referral, and conformal coverage with Wilson intervals, an acceptance-criterion ablation, and an operating-point sensitivity grid;
- source-signature controls (embedding → native-resolution prediction; within-resolution-stratum discrimination);
- robustness over three perceptual-hash radii, five grouped partitioning seeds, and a second frozen encoder (ResNet-50);
- a two-class check of the negative control on the single-source Shenzhen tuberculosis set;
- manuscript-ready tables, figures, predictions, models, a run manifest, and audit exports.

## Principal findings

Results of notebook version R1-1.1 (one execution on a Google Colab Tesla T4 runtime; every value is traceable to a table under `results/R1-1.1/tables/`). The complete results archive of that run, including all figures, per-image audit tables, bootstrap samples, test predictions, thresholds, and serialized models, is attached to the [v1.1 release](https://github.com/hamamh66/Auditing-Before-Automation-CXR-Triage/releases/tag/v1.1) as `Auditing_Before_Automation_CXR_Triage_R1_results.zip`:

| Finding | Result |
|---|---:|
| Source JPEG files | 25,553 |
| Files in contradictory exact identities | 11,176 (43.74%) |
| Duplicate-controlled cohort | 14,281 |
| Internal test images | 1,436 |
| Direct classifier macro-F1 (95% CI) | 0.9867 (0.9806–0.9924) |
| Hierarchical classifier macro-F1 (95% CI) | 0.9855 (0.9786–0.9916) |
| Metadata-only macro-F1, seven variables (95% CI) | 0.9811 (0.9736–0.9879) |
| Metadata-only macro-F1, native dimensions only | 0.9795 |
| Metadata-only macro-F1, intensity statistics only | 0.5765 |
| Paired difference, direct − metadata (95% CI) | 0.0055 (−0.0036 to 0.0150) |
| Selective-policy coverage | 75.42% (1,083 accepted, 0 observed errors) |
| Conformal marginal coverage | 91.50% at a 90% target (117 empty sets) |
| Class–resolution association (Cramér's V) before → after exclusion | 0.80 → 0.99 |

The paired bootstrap interval for the hierarchical-minus-direct macro-F1 difference included zero, and so did the interval for the image-minus-metadata difference. The shortcut is almost entirely the native image size, the frozen embedding predicts the native resolution stratum of a resized image with 0.87 accuracy, and the conclusion holds across five partitioning seeds, two encoders, and three hash radii. On the single-source Shenzhen set the dimensions-only control falls to 0.68 while the image model retains a margin over the metadata control. The reported image-model scores must therefore be interpreted as internal behavior within a confounded public compilation, not as estimates of clinical diagnostic effectiveness.

## Repository contents

```text
.
├── Auditing_Before_Automation_CXR_Triage_R1.ipynb   # current workflow (version R1-1.1)
├── Auditing_Before_Automation_CXR_Triage.ipynb      # original submitted workflow (provenance)
├── requirements.txt                                 # dependencies for a local environment
├── results/
│   └── R1-1.1/                                      # outputs of the reported R1-1.1 run
│       ├── tables/                                  # CSV tables (split assignments, per-image audits, bootstrap samples and test predictions are in the release archive)
│       ├── run_manifest.json                        # library versions, device, full configuration
│       └── outputs_summary.txt                      # run summary
├── Outputs/                                         # figures and tables from an earlier
│   ├── Figures/                                     #   exploratory run (see the note below)
│   └── Tables/
├── LICENSE
└── README.md
```

> **Note on `Outputs/`.** The files under `Outputs/` come from an earlier
> exploratory run — a two-class (normal / pneumonia) model comparison including
> ResNet-18 and EfficientNet-B0 baselines — and do **not** correspond to the
> three-class case study reported in the manuscript. They are kept for
> provenance only. The reported evidence is under `results/R1-1.1/`; running the
> R1 notebook regenerates it and writes its own output tree, described below.

The notebook is self-contained and automatically creates the following output structure:

```text
Auditing_Before_Automation_CXR_Triage_R1/
├── figures/
├── tables/
├── raw/
├── models/
└── outputs_summary.txt
```

At completion, these outputs are compressed as:

```text
/content/Auditing_Before_Automation_CXR_Triage_R1_results.zip
```

## Local execution

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab Auditing_Before_Automation_CXR_Triage_R1.ipynb
```

A CUDA device is recommended for the feature-extraction pass; the audit stages
run on CPU.

## Quick start in Google Colab

1. Open `Auditing_Before_Automation_CXR_Triage_R1.ipynb` in Google Colab.
2. Select a GPU runtime: **Runtime → Change runtime type → T4 GPU**.
3. Run all cells in order.
4. If the dataset is absent, the notebook downloads it automatically from Kaggle.
5. Download the generated results ZIP from the Colab file panel or automatic download prompt.

The GPU-intensive stages are the frozen feature-extraction passes (one per encoder). Downstream classifiers operate on cached embeddings, so the complete comparison, including the robustness study, does not require repeated convolutional-network training. The complete R1 run, including dataset download, takes about 50 minutes on a T4 runtime. Section 16 (Shenzhen) downloads a second public dataset from Kaggle; set `CONFIG["run_shenzhen"] = False` to skip it.

## Dataset

The main workflow uses the public [Chest X-Ray Dataset on Kaggle](https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset), distributed in three folder classes:

- `normal`
- `pneumonia`
- `tuberculosis`

The provider-supplied train, validation, and test folders are not accepted as experimental partitions. All files are reintegrated, audited, grouped by image identity, and repartitioned without allowing an exact or perceptual duplicate group to cross partitions.

The second-dataset check uses the Shenzhen tuberculosis chest-radiograph set (Jaeger et al., IEEE TMI 2014), obtained from its public [Kaggle mirror](https://www.kaggle.com/datasets/kmader/pulmonary-chest-xray-abnormalities).

The original radiographs are not redistributed in this repository. Dataset access remains subject to the providers' current terms.

## Experimental protocol

The workflow enforces four distinct partitions:

| Partition | Purpose |
|---|---|
| Training | Fit direct and hierarchical classifiers |
| Validation | Select the direct model and estimate temperatures |
| Calibration | Estimate conformal quantiles and operating thresholds |
| Internal test | Perform the final locked retrospective evaluation |

The test partition is accessed only after model selection, probability calibration, conformal calibration, and selective-policy thresholds have been fixed.

## Reproducibility

The primary configuration is declared near the beginning of the notebook. Important settings include:

```python
CONFIG = {
    "notebook_version": "R1-1.1",
    "deterministic": True,            # full-precision, deterministic feature extraction
    "seed": 20260815,
    "partition_seeds": [20260815, 20260816, 20260817, 20260818, 20260819],
    "image_size": 160,
    "encoder": "mobilenetv3_small_100",
    "encoders": {"mobilenetv3_small_100": "MobileNetV3-Small", "resnet50": "ResNet-50"},
    "near_duplicate_hamming": 4,
    "hash_radii": [2, 4, 8],
    "bootstrap_replicates": 2000,
    "conformal_alpha": 0.10,
    "target_auto_coverage": 0.75,
    "target_abnormal_sensitivity": 0.98,
    "target_tb_sensitivity": 0.95,
}
```

Feature extraction runs in full 32-bit precision with deterministic cuDNN/cuBLAS kernels, so repeated runs on the same hardware class yield identical embeddings. The original submitted notebook used mixed-precision inference with cuDNN autotuning, which is not bitwise reproducible between runs; this is why its direct-model figures differ slightly from the R1-1.1 values above.

The notebook exports split assignments, image-audit rows, excluded conflicts, test probabilities, predictions, confidence intervals, conformal summaries, referral behavior, figures, and serialized models.

## Evidence boundary

This repository supports a reproducible computational case study, not a clinically validated diagnostic system.

- Patient identifiers and complete acquisition-source identifiers are unavailable.
- Patient-level and institution-level independence cannot be verified.
- The test set is reconstructed from the same public compilation and is not external.
- Robustness to partitioning seed, frozen encoder, and perceptual-hash radius has been assessed internally only.
- The selective thresholds have not been prospectively validated.
- The metadata control reveals strong class–source confounding.

Independent, patient-indexed, source-aware external cohorts are required before clinical-performance, fairness, or deployment claims can be considered.

## Responsible use

The code is intended for research, education, dataset auditing, and reproducibility studies. It must not be used for autonomous diagnosis or clinical decision-making. Predictions require qualified clinical interpretation and independent validation under the intended conditions of use.

## Citation

If this repository supports your work, please cite the accompanying manuscript. Formal journal citation details can be added here after publication.

```bibtex
@article{Mallek2026AuditingBeforeAutomation,
  title   = {Auditing Before Automation: Shortcut-Aware Hierarchical Selective Triage of Pneumonia and Tuberculosis on Chest Radiographs},
  author  = {Mallek, Fatma and Zayoud, Rahma and Hamam, Habib},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## Contact

For questions about the study or repository, please contact the corresponding author through the contact information provided in the manuscript.
