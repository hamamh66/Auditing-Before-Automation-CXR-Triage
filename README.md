# Auditing Before Automation: Shortcut-Aware CXR Triage

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](Auditing_Before_Automation_CXR_Triage.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hamamh66/Auditing-Before-Automation-CXR-Triage/blob/main/Auditing_Before_Automation_CXR_Triage.ipynb)

This repository contains the complete computational workflow supporting the manuscript:

> **Auditing Before Automation: Shortcut-Aware Hierarchical Selective Triage of Pneumonia and Tuberculosis on Chest Radiographs**  
> Fatma Mallek, Rahma Zayoud, and Habib Hamam

The project studies three-class chest-radiograph classification—normal, pneumonia, and tuberculosis—while treating data integrity and shortcut detection as prerequisites for any automation claim. Its central lesson is deliberately cautious: strong internal discrimination is not evidence of clinical generalization when simple acquisition metadata can reproduce nearly the same result.

## Why this repository matters

Public medical-image compilations can contain duplicated images, contradictory labels, and class-linked acquisition signatures. A model can therefore appear highly accurate while learning dataset provenance rather than pathology. This workflow places auditing before model interpretation and lets a metadata-only negative control constrain the scientific conclusion.

The notebook integrates:

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
- a seven-variable metadata-only shortcut control;
- manuscript-ready tables, figures, predictions, models, and audit exports.

## Principal findings

The supplied manuscript reports the following internal case-study results:

| Finding | Result |
|---|---:|
| Source JPEG files | 25,553 |
| Files in contradictory exact identities | 11,176 (43.74%) |
| Duplicate-controlled cohort | 14,281 |
| Internal test images | 1,436 |
| Direct classifier macro-F1 | 0.9838 |
| Hierarchical classifier macro-F1 | 0.9839 |
| Selective-policy coverage | 75.21% |
| Observed errors among accepted cases | 0 |
| Conformal marginal coverage | 91.16% at a 90% target |
| Metadata-only macro-F1 | 0.9811 |

The paired bootstrap interval for the hierarchical-minus-direct macro-F1 difference included zero. More importantly, the metadata-only control nearly matched both image classifiers. The reported image-model scores must therefore be interpreted as internal behavior within a confounded public compilation, not as estimates of clinical diagnostic effectiveness.

## Repository contents

```text
.
├── Auditing_Before_Automation_CXR_Triage.ipynb
└── README.md
```

The notebook is self-contained and automatically creates the following output structure:

```text
Auditing_Before_Automation_CXR_Triage/
├── figures/
├── tables/
├── raw/
├── models/
└── outputs_summary.txt
```

At completion, these outputs are compressed as:

```text
/content/Auditing_Before_Automation_CXR_Triage_results.zip
```

## Quick start in Google Colab

1. Open `Auditing_Before_Automation_CXR_Triage.ipynb` in Google Colab.
2. Select a GPU runtime: **Runtime → Change runtime type → T4 GPU**.
3. Run all cells in order.
4. If the dataset is absent, the notebook downloads it automatically from Kaggle.
5. Download the generated results ZIP from the Colab file panel or automatic download prompt.

The only GPU-intensive stage is a single frozen feature-extraction pass. Downstream classifiers operate on cached embeddings, so the complete comparison does not require repeated convolutional-network training.

## Dataset

The workflow uses the public [Chest X-Ray Dataset on Kaggle](https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset), distributed in three folder classes:

- `normal`
- `pneumonia`
- `tuberculosis`

The provider-supplied train, validation, and test folders are not accepted as experimental partitions. All files are reintegrated, audited, grouped by image identity, and repartitioned without allowing an exact or perceptual duplicate group to cross partitions.

The original radiographs are not redistributed in this repository. Dataset access remains subject to the provider's current terms.

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
    "seed": 20260815,
    "image_size": 160,
    "encoder": "mobilenetv3_small_100",
    "near_duplicate_hamming": 4,
    "bootstrap_replicates": 500,
    "conformal_alpha": 0.10,
    "target_auto_coverage": 0.75,
    "target_abnormal_sensitivity": 0.98,
    "target_tb_sensitivity": 0.95,
}
```

The notebook exports split assignments, image-audit rows, excluded conflicts, test probabilities, predictions, confidence intervals, conformal summaries, referral behavior, figures, and serialized models.

## Evidence boundary

This repository supports a reproducible computational case study, not a clinically validated diagnostic system.

- Patient identifiers and complete acquisition-source identifiers are unavailable.
- Patient-level and institution-level independence cannot be verified.
- The test set is reconstructed from the same public compilation and is not external.
- One frozen encoder, one primary partitioning seed, and one perceptual-hash radius are reported.
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
