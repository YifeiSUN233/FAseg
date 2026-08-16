# scripts/

This directory contains the Jupyter notebooks that implement the main
experimental procedures of FAseg.

## 1. Ablation Experiments

The five pretraining model variants correspond to the five `pretrain_*.ipynb`
notebooks:

| File | Model variant | Training configuration |
| --- | --- | --- |
| `pretrain_stage_full.ipynb` | Full | Three-stage curriculum: Stage 1 basic training (20 epochs) → Stage 2 + TOF filter (40 epochs) → Stage 3 + domain randomization (40 epochs) |
| `pretrain_stage_no_aug.ipynb` | No augmentation | Single 100-epoch stage using only the Stage-1 basic input (TOF filter and domain randomization disabled) |
| `pretrain_stage_no_tof.ipynb` | No TOF filter | Keeps the three-stage schedule but sets `filt_strength = 1.0`, making the TOF filter an identity (no-op) |
| `pretrain_stage_no_domainrand.ipynb` | No domain randomization | Removes the Stage-3 domain randomization: Stage 1 (20 epochs) → Stage 2 TOF filter (80 epochs) |
| `pretrain_stage_no_warmup.ipynb` | No warm-up | Removes the Stage-1 basic warm-up: Stage 2 TOF filter (60 epochs) → Stage 3 domain randomization (40 epochs) |

> Note: the "TOF filter" in the code (`TOF_filt_enabled` / `filt_strength`)
> corresponds to the Stage-2 correction of the overall amplitude over a
> partial region of the FA-region described in the paper.

- `ablation_inference_on_ex_vivo_data.ipynb` — runs inference on the ex-vivo
  data for each ablation model.
- `ablation_metrics_on_ex_vivo_data.ipynb` — reports the metrics of the
  inference results above.

## 2. Inference Comparison Before / After Fine-tuning

- `run_finetuning_ex_vivo.ipynb` / `run_finetuning_in_vivo.ipynb` — fine-tune
  the pretrained model (the `best_s3` checkpoint) on the ex-vivo / in-vivo
  data, respectively.
- `run_inference_ex_vivo.ipynb` / `run_inference_in_vivo.ipynb` — run
  inference with the fine-tuned models.
- `metrics_finetuning_comparison.ipynb` — compares the metrics of the
  corresponding inference results.

> Note: to protect volunteer privacy, the raw RF signals of the in-vivo data
> are **not** distributed in this repository. As a result, the in-vivo
> fine-tuning and inference notebooks (`run_finetuning_in_vivo.ipynb`,
> `run_inference_in_vivo.ipynb`) cannot run as-is. The corresponding inference
> results are nevertheless provided, so `metrics_finetuning_comparison.ipynb`
> runs normally.

## 3. Pretraining Input Samples

- `sample_of_datasets.ipynb` — visualizes the differences among the
  Stage-1/2/3 input samples of the pretraining stage (signal / mask / overlay
  of one fixed sample under the three configurations), and can export the
  results to a `.mat` file.
