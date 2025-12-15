# MLP-Mixer Tiny on CIFAR-10

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Parameters-<1M-blue" alt="Parameters">
</p>

A PyTorch implementation of a tiny MLP-Mixer model (<1M parameters) trained on CIFAR-10 with modern training techniques including EMA, Mixup, AutoAugment, and torch.compile optimization.

## 📋 Overview

This repository implements a lightweight MLP-Mixer architecture that achieves competitive performance on CIFAR-10 with fewer than 1 million parameters. The implementation includes several modern training enhancements:

- **MLP-Mixer Architecture**: Token-mixing and channel-mixing MLPs instead of self-attention
- **Training Optimizations**: Exponential Moving Average (EMA), Mixup data augmentation
- **Data Augmentation**: AutoAugment policy for CIFAR-10, random crops, horizontal flips
- **Performance**: TF32 precision support, torch.compile integration for faster training
- **Metrics**: Comprehensive evaluation including per-class precision, recall, F1 scores

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision numpy tqdm
# Optional for visualization and metrics:
pip install matplotlib seaborn scikit-learn
```

### Training

```bash
python train_mlpmixer.py
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--epochs` | Number of training epochs | 200 |
| `--batch-size` | Training batch size | 128 |
| `--lr` | Learning rate | 1e-3 |
| `--weight-decay` | Weight decay for AdamW | 1e-4 |
| `--mixup` | Enable Mixup augmentation | False |
| `--mixup-alpha` | Mixup alpha parameter | 0.8 |
| `--seed` | Random seed | 42 |
| `--save-dir` | Directory for checkpoints | ./checkpoints |
| `--resume` | Path to checkpoint to resume from | "" |
| `--device` | Device to use (cuda/cpu) | cuda if available |

Example with custom settings:
```bash
python train_mlpmixer.py --epochs 100 --batch-size 256 --mixup --mixup-alpha 1.0 --save-dir ./my_checkpoints
```

## 📁 Project Structure

```
.
├── train_mlpmixer.py          # Main training script
├── README.md                  # This file
├── checkpoints/               # Saved models and metrics
│   ├── cifar10_best.pt       # Best model checkpoint
│   ├── cifar10_final_metrics.txt  # Detailed metrics
│   └── cifar10_confusion_matrix.png  # Visualization
└── data/                     # CIFAR-10 dataset (auto-downloaded)
```

## 🏗️ Model Architecture

The MLP-Mixer Tiny model consists of:

1. **Patch Embedding**: Splits 32×32 images into 4×4 patches (64 patches total)
2. **Mixer Blocks**: 6 alternating token-mixing and channel-mixing MLPs
   - Token-mixing MLP: Operates across patches
   - Channel-mixing MLP: Operates across channels
3. **Classification Head**: Global average pooling + linear layer

**Key Specifications:**
- Embedding dimension: 160
- Token MLP hidden dim: 64 (1.0× num_patches)
- Channel MLP hidden dim: 480 (3.0× embed_dim)
- Total parameters: ~950K (< 1M)

## 🛠️ Features

### Training Enhancements

- **Exponential Moving Average (EMA)**: Stable training with momentum 0.9999
- **Mixup Augmentation**: Optionally blend images and labels for regularization
- **Cosine Annealing with Warmup**: 10-epoch warmup, then cosine decay
- **AdamW Optimizer**: With weight decay 1e-4

### Data Augmentation

- Random cropping (32×32 with padding 4)
- Random horizontal flipping
- AutoAugment (CIFAR-10 policy)
- Normalization with CIFAR-10 statistics

### Performance Optimizations

- **TF32 Precision**: Automatic enabling on Ampere+ GPUs
- **torch.compile**: Reduced overhead mode for faster training
- **Non-blocking data loading**: With pin_memory for GPU transfers
- **Gradient checkpointing**: Memory-efficient training

## 📊 Results

Typical performance on CIFAR-10 test set:
- **Test Accuracy**: ~87-88%
- **Training Time**: ~2-3 hours on a single GPU
- **Memory Usage**: < 2GB VRAM with batch size 128

## 📈 Output Files

The training script generates:

1. **Best Model Checkpoint** (`cifar10_best.pt`):
   - Model weights
   - Optimizer and scheduler states
   - Training arguments
   - Best validation accuracy

2. **Metrics File** (`cifar10_final_metrics.txt`):
   - Overall accuracy
   - Macro-average precision, recall, F1
   - Per-class metrics for all 10 CIFAR-10 classes
   - Confusion matrix

3. **Visualization** (`cifar10_confusion_matrix.png`):
   - Heatmap of confusion matrix
   - Generated if matplotlib/seaborn are available

## 🔧 Advanced Usage

### Resume Training

```bash
python train_mlpmixer.py --resume ./checkpoints/cifar10_best.pt --epochs 250
```

### CPU-Only Training

```bash
python train_mlpmixer.py --device cpu --batch-size 64
```

### Custom Model Parameters

Modify the `MLPMixerTiny` initialization in `main()`:
```python
model = MLPMixerTiny(
    img_size=32,
    patch_size=4,
    in_ch=3,
    embed_dim=192,  # Increase embedding dimension
    num_blocks=8,   # More mixer blocks
    token_hidden_mul=1.5,
    channel_hidden_mul=4.0,
    num_classes=10,
    drop=0.1        # Add dropout
).to(device)
```

## ⚠️ Notes & Limitations

1. **GPU Requirements**: TF32 requires Ampere or newer NVIDIA GPUs (RTX 30xx, A100, etc.)
2. **torch.compile**: May fail on some systems; falls back to eager mode gracefully
3. **Reproducibility**: Full reproducibility requires CUDA deterministic mode (slower)
4. **Memory**: Batch size 128 requires ~2GB VRAM; reduce for smaller GPUs

## 🤝 Contributing

Feel free to submit issues and pull requests for:
- Bug fixes
- Performance improvements
- Additional features
- Support for other datasets

## 📚 References

1. [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)
2. [Mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
3. [AutoAugment: Learning Augmentation Strategies from Data](https://arxiv.org/abs/1805.09501)
4. [AdamW: Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- The MLP-Mixer authors at Google Research
- PyTorch team for torch.compile and optimization features
- CIFAR-10 dataset creators
