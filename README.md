# SomnoOrbit

Implementation of **Internalizing Physiological Transition Priors for Contextual Sleep Staging**.

SomnoOrbit combines contextual EEG predictions with a physiological reference orbit. Context Origin Aggregation (COA) averages overlapping predictions; a Reference Sleep Orbit (RSO) represents night position and REM-anchored phase; Evidence Coupled Orbit Deformation (ECOD) adapts the reference using EEG evidence; Reciprocal Uncertainty Readout (RUR) combines EEG and orbit predictions.

## Repository structure

```text
model/             Neural backbones, orbit modules, and training utilities
config/            Default ISRUC and HMC configurations
preprocessing/     Dataset preparation and loading
demo_isruc.py      ISRUC training demo
demo_hmc.py        HMC training demo
requirements.txt   Python dependencies
setup.sh           Environment installation
```

## Installation

Use Python 3.12.

```bash
git clone https://github.com/junye988/SomnoOrbit.git
cd SomnoOrbit
bash setup.sh
source .venv/bin/activate
```

To select a PyTorch wheel index during installation:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 bash setup.sh
```

For a CPU environment, use `https://download.pytorch.org/whl/cpu`.

On Windows, the equivalent PowerShell commands are:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\Activate.ps1
```

## Data availability

The datasets are available from their respective providers:

- **ISRUC-Sleep Group I:** [ISRUC-Sleep dataset downloads](https://sleeptight.isr.uc.pt/?page_id=48). Download and extract the recordings and scoring files for subjects 1--100.
- **Haaglanden Medisch Centrum (HMC), version 1.1:** [PhysioNet dataset page and downloads](https://physionet.org/content/hmc-sleep-staging/1.1/). Download the recordings and sleep-scoring files.

## Preprocessing

### ISRUC-Sleep

Arrange the extracted Group I files as follows:

```text
data/isruc_raw/
  1/
    1.rec
    1_1.txt
  2/
    2.rec
    2_1.txt
  ...
```

```bash
python -m preprocessing.prepare_isruc --raw data/isruc_raw --processed data/isruc_processed --n-jobs 4
```

The output contains paired NumPy sequences in `seq/ISRUC-group1-<subject>/` and `labels/ISRUC-group1-<subject>/`. Each sequence has shape `[20, 6, 6000]`. Signals use the six derivations F3-A2, C3-A2, O1-A2, F4-A1, C4-A1, and O2-A1 at 200 Hz, with a 0.3--35 Hz band-pass filter and a 50 Hz notch filter. Labels come from the first scorer. Incomplete final contexts are discarded.

### HMC

Keep each recording with its sleep-scoring files in the extracted dataset directory:

```text
data/hmc_raw/
  SN001.edf
  SN001_sleepscoring.edf
  SN001_sleepscoring.txt
  ...
```

```bash
python -m preprocessing.prepare_hmc --raw data/hmc_raw --processed data/hmc_processed --n_jobs 4
```

The output contains `train/`, `eval/`, `test/`, and `meta_summary.pkl`. Each epoch stores four derivations (F4-M1, C4-M1, O2-M1, C3-M2) with shape `[4, 3000]`. Signals are band-pass filtered at 0.3--45 Hz, notch filtered at 50 Hz, and resampled to 100 Hz. The default split seed is 42.

Both datasets use 30-second epochs and the label order `W, N1, N2, N3, REM` (`0, 1, 2, 3, 4`). During training and inference, each 20-epoch context is normalized per channel using its median and median absolute deviation, then clipped to `[-8, 8]`.

## Training

### Quick demos

Each quick demo runs one training epoch on a small data subset, estimates the orbit templates, evaluates 60 epochs from a validation recording, and saves the trained model.

```bash
python demo_isruc.py --data-root data/isruc_processed --quick
python demo_hmc.py --data-root data/hmc_processed --quick
```

### Full training

```bash
python demo_isruc.py --data-root data/isruc_processed --output-dir outputs/isruc_full
python demo_hmc.py --data-root data/hmc_processed --output-dir outputs/hmc_full
```

The demos use CNN--BiGRU with width 64 and contexts of 20 epochs. The default training configuration uses AdamW (`lr=3e-4`, `weight_decay=1e-3`), batch size 8, up to 18 epochs, early-stopping patience 6, and class-balanced cross-entropy plus a trajectory-energy term weighted by 0.05. Checkpoints are selected by validation balanced accuracy. Orbit templates are estimated from training recordings. Context aggregation uses stride 5 for template estimation and stride 2 for validation.

Default settings are in `config/isruc.json` and `config/hmc.json`. The ISRUC subject split uses seed 2026 and assigns 70 subjects to training, 10 to validation, and 20 to testing. HMC uses its preprocessed recording split.

Common options:

| Option | Purpose |
|---|---|
| `--config PATH` | Load a JSON configuration |
| `--output-dir PATH` | Select the output directory |
| `--device auto` | Use CUDA when available, otherwise CPU |
| `--device cpu` | Run on CPU |
| `--device cuda:0` | Select a CUDA device |
| `--epochs N` | Set the training epoch limit |
| `--workers N` | Set the number of data-loading workers |
| `--seed N` | Set the model and training seed |

Each run writes:

- `backbone.pt`: selected EEG backbone checkpoint;
- `somnoorbit.pt`: backbone, orbit templates, and orbit configuration;
- `config.json`: run settings;
- `history.json`: training loss and validation metrics by epoch;
- `metrics.json`: validation metrics for EEG evidence and SomnoOrbit.

## Inference

```python
from model.checkpoint import load_model
from preprocessing.isruc_data import load_night

model, checkpoint = load_model("outputs/isruc_full/somnoorbit.pt", device="cpu")
night, labels = load_night("data/isruc_processed", subject=1)
predictions = model.predict(night, context=20, stride=2, batch_size=8)
```

`night` has shape `[epochs, channels, samples]`. `predictions` contains one stage index per epoch.

## Acknowledgments

ISRUC preprocessing incorporates code from [CBraMod](https://github.com/wjq-learning/CBraMod) and its EDF reader based on [MNE-Python](https://mne.tools/). Their license notices are included in `preprocessing/`.
