# Auditing Before Automation: Shortcut-Aware CXR Triage

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](Auditing_Before_Automation_CXR_Triage.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hamamh66/Auditing-Before-Automation-CXR-Triage/blob/main/Auditing_Before_Automation_CXR_Triage.ipynb)

This repository contains the complete computational workflow supporting the article:

> **Auditing Before Automation: Shortcut-Aware Hierarchical Selective Triage of Pneumonia and Tuberculosis on Chest Radiographs**  
> Fatma Mallek, Rahma Zayoud, and Habib Hamam

The workflow studies three-class chest-radiograph classification—normal, pneumonia, and tuberculosis—while treating data integrity and shortcut detection as prerequisites for any automation claim. Its central lesson is deliberately cautious: strong internal discrimination is not evidence of clinical generalization when the native image dimensions alone reproduce nearly the same result.

## What the notebook does

`Auditing_Before_Automation_CXR_Triage.ipynb` is a single, self-contained notebook. It integrates:

- SHA-256 exact-identity auditing and removal of identities carrying contradictory labels;
- 64-bit perceptual hashing with BK-tree search, grouping images into **candidate near-duplicate** components (visual similarity, not verified identity);
- duplicate-group-aware train/validation/calibration/test partitioning;
- frozen MobileNetV3-Small and ResNet-50 feature extraction (one pass per encoder);
- direct multiclass and hierarchical classification;
- validation-only temperature scaling;
- paired stratified bootstrap and cluster bootstrap confidence intervals;
- class-conditional conformal prediction and a calibration-locked selective-referral policy;
- metadata negative controls: **native dimensions only (primary control)**, intensity statistics only, and all seven variables;
- an exploratory **evidence-gate rule** applied to every configuration: the gate is triggered when the lower 95% bound of the paired image-minus-control macro-F1 difference does not exceed a margin (zero by default) or when the control-to-image ratio exceeds 0.95;
- forensic characterization of the excluded files;
- class-wise acceptance, referral, and coverage; acceptance-criterion ablation; operating-point sensitivity;
- source-signature controls (embedding → native-resolution prediction; within-resolution-stratum discrimination);
- robustness over three perceptual-hash radii, five grouped partitioning seeds, and two frozen encoders;
- a chaining diagnostic for the perceptual components, with de-identified visual examples per radius;
- a two-class check on the single-source Shenzhen tuberculosis set with component-grouped cross-validation, identical folds for all models, and paired cluster-bootstrap intervals.

Every long-running step prints its progress. Small result files (tables, figures, JSON summaries) are mirrored to Google Drive when the notebook runs in Colab; datasets, embeddings, and models stay on the runtime disk.

## Principal findings

All values come from a deterministic CPU execution of the notebook; each is traceable to a table under `Results/Tables/`. The complete results archive (all figures, per-image audit tables, split assignments, bootstrap samples, test predictions, thresholds, and serialized models) is attached to the latest release.

| Finding | Result |
|---|---:|
| Source JPEG files | 25,553 |
| Files in contradictory exact identities | 11,176 (43.74%) |
| Identity-conflict-controlled cohort | 14,281 |
| Internal test images | 1,436 |
| Direct classifier macro-F1 (95% CI) | 0.9859 (0.9796–0.9918) |
| Hierarchical classifier macro-F1 (95% CI) | 0.9855 (0.9785–0.9916) |
| Dimensions-only control macro-F1 (95% CI) | 0.9795 (0.9716–0.9864) |
| Seven-variable control macro-F1 (95% CI) | 0.9811 (0.9736–0.9879) |
| Intensity-statistics-only control macro-F1 | 0.5765 |
| Paired difference, direct − dimensions (95% CI) | 0.0065 (−0.0033 to 0.0164) |
| Evidence gate on the compilation | triggered in all 12 configurations |
| Selective-policy coverage | 75.42% (1,083 accepted, 0 observed errors; 0–2 errors across configurations) |
| Conformal marginal coverage | 91.50% at a 90% target (117 empty sets) |
| Class–resolution association (Cramér's V) before → after exclusion | 0.80 → 0.99 |
| Shenzhen: image minus dimensions-only control | +0.168 (MobileNetV3-Small), +0.142 (ResNet-50); gate not triggered |

The dimensions alone reproduce the image-model performance on the compilation, and in three configurations they outperform it. On the single-source Shenzhen set the dimension signal is much weaker and the image representation retains a clear margin, so the gate is not triggered. The image-model scores on the compilation must therefore be read as internal behavior within a public compilation whose labels are confounded with acquisition or export characteristics, not as estimates of clinical diagnostic effectiveness.

## Repository contents

```text
.
├── Auditing_Before_Automation_CXR_Triage.ipynb   # complete workflow
├── requirements.txt                              # dependencies for a local environment
├── Results/
│   ├── Tables/                                   # CSV tables of the reported run
│   ├── run_manifest.json                         # library versions, device, full configuration
│   ├── integrity_audit.json, shenzhen_*.json, temperatures.json, test_component_summary.json
│   └── outputs_summary.txt                       # run summary
├── LICENSE
└── README.md
```

Large per-image tables (image audit, split assignments, excluded conflicts, bootstrap samples, and test predictions), all figures, and the serialized models are in the results archive attached to the latest release.

When run, the notebook writes:

```text
Auditing_Before_Automation_CXR_Triage/
├── figures/
├── tables/
├── raw/
├── models/
└── outputs_summary.txt
```

and compresses it to `Auditing_Before_Automation_CXR_Triage_results.zip`.

## Quick start in Google Colab

1. Open the notebook in Google Colab (badge above).
2. A GPU runtime (T4) shortens feature extraction; the notebook also runs on a CPU-only runtime.
3. Run all cells in order. Both datasets are downloaded automatically from Kaggle if absent.
4. If the runtime disconnects, run all cells again: cached audit tables and embeddings are reused, and the robustness study resumes from its checkpoint.

The reported run was executed on a CPU-only runtime and took about 50 minutes of computation in total. Set `CONFIG["run_shenzhen"] = False` to skip the second-dataset check.

## Local execution

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab Auditing_Before_Automation_CXR_Triage.ipynb
```

Adjust `CONFIG["dataset_root"]`, `CONFIG["shenzhen_root"]`, and `CONFIG["output_root"]` to local paths.

## Datasets

The main workflow uses the public [Chest X-Ray Dataset on Kaggle](https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset), distributed in three folder classes (`normal`, `pneumonia`, `tuberculosis`). The provider-supplied train, validation, and test folders are not used as experimental partitions: all files are audited, grouped by image identity, and repartitioned so that no exact or candidate near-duplicate group crosses partitions.

The second-dataset check uses the Shenzhen tuberculosis chest-radiograph set (Jaeger et al., IEEE Transactions on Medical Imaging, 2014), obtained from its public [Kaggle mirror](https://www.kaggle.com/datasets/kmader/pulmonary-chest-xray-abnormalities).

The radiographs are not redistributed in this repository. Dataset access remains subject to the providers' current terms.

## Experimental protocol

| Partition | Purpose |
|---|---|
| Training | Fit direct and hierarchical classifiers and the metadata controls |
| Validation | Select the direct model and estimate temperatures |
| Calibration | Estimate conformal quantiles and operating thresholds |
| Internal test | Final locked retrospective evaluation |

The test partition is accessed only after model selection, probability calibration, conformal calibration, and selective-policy thresholds have been fixed.

## Reproducibility

The configuration is declared at the beginning of the notebook. Key settings:

```python
CONFIG = {
    "deterministic": True,
    "seed": 20260815,
    "partition_seeds": [20260815, 20260816, 20260817, 20260818, 20260819],
    "image_size": 160,
    "encoders": {"mobilenetv3_small_100": "MobileNetV3-Small", "resnet50": "ResNet-50"},
    "near_duplicate_hamming": 4,
    "hash_radii": [2, 4, 8],
    "bootstrap_replicates": 2000,
    "conformal_alpha": 0.10,
    "target_auto_coverage": 0.75,
    "target_abnormal_sensitivity": 0.98,
    "target_tb_sensitivity": 0.95,
    "gate_margin": 0.0,
    "gate_ratio": 0.95,
}
```

Feature extraction runs in full 32-bit precision with deterministic algorithms, so repeated runs on the same hardware class yield identical embeddings. Results obtained on different hardware (for example GPU instead of CPU) can differ in the third or fourth decimal place.

## Evidence boundary

This repository supports a reproducible computational case study, not a clinically validated diagnostic system.

- Patient identifiers and acquisition-source identifiers are unavailable; patient-level and institution-level independence cannot be verified.
- Removing conflicting identities controls label conflicts; it does not show that the remaining folder labels are diagnostically correct.
- Perceptual-hash groups are candidate near-duplicates; they can join distinct radiographs.
- The test set is reconstructed from the same public compilation and is not external.
- Bootstrap intervals are conditional on the fitted models.
- The evidence gate is an exploratory rule; a gate that is not triggered does not prove the absence of shortcuts.

## Responsible use

The code is intended for research, education, dataset auditing, and reproducibility studies. It must not be used for autonomous diagnosis or clinical decision-making.

## Citation

```bibtex
@article{Mallek2026AuditingBeforeAutomation,
  title   = {Auditing Before Automation: Shortcut-Aware Hierarchical Selective Triage of Pneumonia and Tuberculosis on Chest Radiographs},
  author  = {Mallek, Fatma and Zayoud, Rahma and Hamam, Habib},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## Contact

For questions about the study or repository, please contact the corresponding author through the contact information provided in the article.
