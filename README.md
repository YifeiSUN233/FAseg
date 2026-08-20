# FBseg — First-Break (FB) Time Segmentation

A 3-stage curriculum U-Net training pipeline for first-break (FB) time picking
from musculoskeletal ultrasound RF signals. It follows a simulation-to-real (Sim2Real)
workflow: the model is first pretrained on synthetic RF data, and then separately 
fine-tuned and evaluated on ex vivo bovine tissue data and in vivo human thigh
measurements.

**Terminology note:** This repository uses *first break (FB)* as the primary term. For compatibility with earlier versions of the code, *first arrival (FA)* is retained in some package names, file paths, variables, and comments. In this repository, FA and FB refer to the same target arrival-picking task.

## Overview

The pipeline trains a 2D U-Net to localize the first-arrival boundary of
the limb in TOF ultrasound signals, using a three-stage curriculum:

| Stage | Curriculum | Purpose |
| --- | --- | --- |
| 1 | Basic synthetic training | Learn the FA boundary from clean simulated RF |
| 2 | + TOF filter | Correct the overall amplitude over a partial region of the FA-region |
| 3 | + domain randomization | Close the synthetic-to-real domain gap |

## Repository structure

```
FBseg/
├── README.md
├── environment.yml                  # Conda environment
├── .gitignore
├── datasets/                        # Demonstration subset of the pretraining data
│   └── README.md                    #   (full database described there)
├── manual segmentation/             # Manual annotations for the real data
├── outputs/                         # Pretraining checkpoints + inference results
│   └── README.md
├── scripts/                         # Jupyter notebooks implementing the experiments
│   └── README.md
└── src/
    └── faseg/
        ├── data/
        │   ├── dataset.py           # LimbArrayDataset, RealLimbDataset, RealLimbDataset2
        │   └── filt_max_steps_vec.csv  # TOF-filter calibration
        └── models/
            ├── unet.py              # UNet / UNet2 architectures
            └── losses.py            # cross-entropy (+ total-variation) losses
```

## Getting started

```bash
# 1. Create the conda environment
conda env create -f environment.yml
conda activate faseg_ablation

# 2. Run the notebooks in scripts/ (all experiments are notebook-based)
jupyter notebook scripts/
```

## Workflow

All experiments are implemented as Jupyter notebooks in `scripts/`:

1. **Pretraining** — `pretrain_stage_*.ipynb` (five model variants: full + four
   ablations).
2. **Ablation** — `ablation_inference_on_ex_vivo_data.ipynb` then
   `ablation_metrics_on_ex_vivo_data.ipynb`.
3. **Fine-tuning** — `run_finetuning_ex_vivo.ipynb` / `run_finetuning_in_vivo.ipynb`.
4. **Inference** — `run_inference_ex_vivo.ipynb` / `run_inference_in_vivo.ipynb`.
5. **Metrics** — `metrics_finetuning_comparison.ipynb`.
6. **Input samples** — `sample_of_datasets.ipynb`.

See `scripts/README.md` for details.

## Data

Only a small demonstration subset of the pretraining data is included in this
repository (see `datasets/README.md`). The full synthetic database originates
from the 2100 numerical phantoms in
[OpenWaves/leg](https://huggingface.co/OpenWaves).

To protect volunteer privacy, the raw RF signals of the in-vivo data are
**not** distributed. The in-vivo fine-tuning and inference notebooks therefore
cannot run as-is; the corresponding inference results are still provided so
the metrics notebook runs normally (see `scripts/README.md`).

## Pretrained checkpoints

The pretraining checkpoints (full model + four ablations) are listed in
`outputs/README.md`.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file.

## Citation

TODO
