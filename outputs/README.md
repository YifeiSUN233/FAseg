# outputs/

## Directory structure

```
outputs/
├── pretraining/                      # Full model: pretraining + fine-tuning
│   ├── unet_epoch84.pth              # Full pretraining best_s3 (epoch 84)
│   ├── config.txt / loss_log.txt     # training configuration / loss log
│   ├── train_indices.json / val_indices.json   # train / val split
│   ├── finetuning_best_s3/           # in-vivo fine-tuning results
│   └── finetuning_exvivo_best_s3/    # ex-vivo fine-tuning results
├── pretraining_no_aug/               # ablation: no augmentation
├── pretraining_no_domainrand/        # ablation: no domain randomization
├── pretraining_no_tof/               # ablation: no TOF filter
├── pretraining_no_warmup/            # ablation: no warm-up
├── inference_results/                # saved inference results (read by the metrics notebook)
│   ├── exvivo/
│   └── invivo/
└── dataset_sanity_check.png / dataset_diff_maps.png / dataset_sanity_check.mat
                                      # outputs of sample_of_datasets.ipynb
```

## Pretraining checkpoints

| Model variant | Checkpoint | Best epoch |
| --- | --- | --- |
| Full | `pretraining/unet_epoch84.pth` | 84 |
| No augmentation | `pretraining_no_aug/unet_epoch61.pth` | 61 |
| No domain randomization | `pretraining_no_domainrand/unet_epoch70.pth` | 70 |
| No TOF filter | `pretraining_no_tof/unet_epoch92.pth` | 92 |
| No warm-up | `pretraining_no_warmup/unet_epoch87.pth` | 87 |

Each directory also contains `config.txt` (training configuration), `loss_log.txt`
(loss curve), and `train_indices.json` / `val_indices.json` (the data split).
