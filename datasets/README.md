# Datasets

This directory holds the data files consumed by the FAseg pretraining stage.
Because the complete database is very large, only a small demonstration subset
is included in this repository. The remaining data can be obtained from the
sources referenced in each section below.

## Directory layout

```
datasets/
├── digital models/
│   └── mapSOS2Density.m  # MATLAB script: sound-speed → density
├── simulated/           # simulated RF signals (1 demonstration slice)
│   └── dat1_src_*
├── labels/              # first-arrival (FA) labels (1 demonstration slice)
│   └── label.1
├── noise/               # measured noise (8 demonstration files)
│   └── test1_src_*
└── README.md
```

---

## 1. Digital phantoms (`digital models/`)

The digital phantoms used for pretraining originate from the 2100 numerical
phantoms in [OpenWaves/leg](https://huggingface.co/OpenWaves). Pretraining
consumes both a sound-speed (SOS) model and a density model.

The SOS model is downloaded directly from OpenWaves. Because the density model
is derived from the SOS distribution, the (very large) density binaries are not
shipped here; instead this repository provides `mapSOS2Density.m`, a MATLAB
script that converts the SOS distribution into the density distribution.

| Script              | Purpose                                            |
| ------------------- | -------------------------------------------------- |
| `mapSOS2Density.m`  | Maps a sound-speed volume to a density volume (SOS → ρ) |

`mapSOS2Density(sos)` takes a `float32` sound-speed array (units m/s) and
returns the same-size density array (units kg/m³). Sound speeds in
[1400, 3700] m/s are linearly mapped to a per-slice density range whose lower
bound follows N(910, 10²) kg/m³ and whose upper bound is sampled from one of
four bone-density distributions.

---

## 2. Simulated RF data (`simulated/`)

Because the full database is too large to distribute, this repository keeps
the RF data and FA labels of only **one** digital-phantom slice as a
demonstration.

- **Naming**: `datX_src_Y`, where `X` is the slice index and `Y` is the
  emitter index.
- **Shape**: **512 × 876** (receiver × time sample, `dt = 2.4e-7 s`), stored as
  `float32` in C order.

---

## 3. First-arrival labels (`labels/`)

- **Naming**: `label.X`, where `X` is the slice index.
- **Shape**: **256 × 512** (emitter × receiver), stored as `float32` in C
  order. Each row holds the first-arrival boundary times of one emitter to all
  512 receivers.

---

## 4. Noise data (`noise/`)

Only 8 noise files are kept for demonstration.

- **Naming**: `testX_src_Y`, where `X` is the FMC acquisition index and `Y` is
  the emitter index.
- **Shape**: **512 × 4955** (receiver × time step, `dt = 4e-8 s`), stored as
  `int16` in C order.

---

## Array geometry

All signals are produced with a ring array of **512 elements** and **11 cm
radius**. The odd-indexed elements `1, 3, 5, …, 511` are used as transmitters
(256 emitters) and **all 512 elements** receive. During pretraining, each slice
stores the RF of only every 32nd emitter (`1, 33, 65, …, 225`, i.e. 8 emitters
per slice), whereas `label.X` contains the first-arrival times for all 256
emitters.
