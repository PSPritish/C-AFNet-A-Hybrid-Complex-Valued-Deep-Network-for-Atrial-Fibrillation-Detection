# Complex-AFNet: A Hybrid Complex-Valued Deep Network for Atrial Fibrillation Detection

Official implementation of **Complex-AFNet**, a hybrid complex-valued deep neural network for detecting Atrial Fibrillation (AF) from ECG signals using Gramian Angular Field (GAF) representations.

---

## Table of Contents

- [Model Architectures](#model-architectures)
- [Installation](#installation)
- [Dataset](#dataset)
- [Pretrained Weights](#pretrained-weights)
- [Evaluation](#evaluation)
- [Citation](#citation)

---

## Model Architectures

| Model | Description | Type |
|-------|-------------|------|
| `resnet18` | Standard ResNet-18 baseline | Real-valued |
| `r_afnet` | Real-valued AFNet | Real-valued |
| `c_resnet18` | Complex-valued ResNet-18 | Complex-valued |
| `c_afnet` | **Complex AFNet (Proposed)** | Hybrid |
| `inverted_c_afnet` | Inverted Complex AFNet | Hybrid |
| `grouped_c_afnet` | Grouped Complex AFNet (dual-stream with attention) | Hybrid |

---

## Installation

```bash
# Clone repository
git clone https://github.com/PSPritish/C-AFNet-A-Hybrid-Complex-Valued-Deep-Network-for-Atrial-Fibrillation-Detection.git
cd Complex-AFNet

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset

> ** Download:** [Dataset Link](https://www.kaggle.com/datasets/pspkmahali/cafnet)

Organize your data in one of these structures:

**Option 1: With train/test/val splits**
```
data_root/
├── GASF/
│   └── test/
│       └── record_number/
│           ├── segment1_label1.jpeg
│           └── segment2_label0.jpeg
└── GADF/
    └── test/
        └── record_number/
            ├── segment1_label1.jpeg
            └── segment2_label0.jpeg
```

**Option 2: Without splits (set mode=None)**
```
data_root/
├── GASF/
│   └── record_number/
│       ├── segment1_label1.jpeg
│       └── segment2_label0.jpeg
└── GADF/
    └── record_number/
        ├── segment1_label1.jpeg
        └── segment2_label0.jpeg
```

- `label0` = Normal (Non-AF)
- `label1` = Atrial Fibrillation (AF)

---

## Pretrained Weights

> ** Download:** [Pretrained Weights](https://www.kaggle.com/models/pspkmahali/c-afnet-weights)

Place weights in the `weights/` directory:

```
weights/
├── resnet18.pth
├── r_afnet.pth
├── c_resnet18.pth
├── c_afnet.pth           
├── inverted_c_afnet.pth
└── grouped_c_afnet.pth
```

---

## Evaluation

Use `evaluate.py` to reproduce our results:

### Evaluate a Single Model

```bash
# Data directly in GASF/GADF folders (no subfolder)
python evaluate.py --model c_afnet --weights ./weights/c_afnet.pth --data_root ./data

# Data in subfolder (e.g., test, train, val)
python evaluate.py --model c_afnet --weights ./weights/c_afnet.pth --data_root ./data --mode test
```

### Evaluate All Models

```bash
python evaluate.py --all --weights_dir ./weights --data_root ./data
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model to evaluate: `resnet18`, `r_afnet`, `c_resnet18`, `c_afnet`, `inverted_c_afnet`, `grouped_c_afnet` | Required* |
| `--weights` | Path to model weights (.pth) | Required* |
| `--all` | Evaluate all models in `--weights_dir` | False |
| `--weights_dir` | Directory containing model weights | `./weights` |
| `--data_root` | Root directory of dataset | Required |
| `--mode` | Subfolder name (`test`, `train`, `val`) or `None` if data is directly in GASF/GADF | `None` |
| `--dataset_type` | `combined`, `gasf`, `gadf`, `complex` | `combined` |
| `--input_channels` | Number of input channels | `3` |
| `--batch_size` | Batch size | `32` |

*Required when not using `--all`


---

## License

MIT License - see [LICENSE](LICENSE) for details.

---
