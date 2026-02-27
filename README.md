
## Overview
An end-to-end comparative study benchmarking traditional from-scratch CNN architectures against modern Transfer Learning and state-of-the-art Vision Transformers (ViT) on the CIFAR-100 dataset. This project explores model capacity, generalization trade-offs, and the challenges of low-resolution image classification.

## Methodology & Training Setup
- **Dataset**: [CIFAR-100](https://huggingface.co/datasets/cifar100) (100 classes, 50k train / 10k test).
- **Adaptive Pipeline**: Upscaled 32×32 inputs to 224×224 (via custom Keras layers and Hugging Face `ViTImageProcessor`) to satisfy standard ImageNet-style structural requirements.
- **Optimization Strategies**: 
  - **16-bit Mixed Precision** to optimize memory footprint and training speed.
  - **Dynamic Learning Rate** (`ReduceLROnPlateau`) and **Early Stopping** to manage convergence.
- **Architectures Evaluated**:
  1. **Transformers (SOTA)**: `ViT` (via Hugging Face Ecosystem).
  2. **Transfer Learning**: `EfficientNetV2` (Pretrained feature extractor).
  3. **From Scratch (Baselines)**: `ResNet-18`, `VGG-13`, `AlexNet`.

## Results & Benchmarks

Models were evaluated based on their Top-1 Evaluation/Validation Accuracy. 

| Strategy | Architecture | Best Eval/Val Accuracy |
| :--- | :--- | :---: |
| Transformers | **Vision Transformer (ViT)** | **90.54%** |
| Transfer Learning| **EfficientNetV2** | **77.39%** |
| From Scratch | ResNet-18 | 63.16% |
| From Scratch | VGG-13 | 57.03% |
| From Scratch | AlexNet | 56.97% |

### Key Insights
High-capacity CNNs (ResNet/VGG) trained from scratch on low-resolution data suffer from severe generalization gaps, tending to fit the training set too well. 

While **Transfer Learning** (EfficientNetV2) provides a strong inductive bias to mitigate overfitting and boosts accuracy to ~77%, migrating to a **self-attention mechanism (ViT)** fundamentally shifts the performance ceiling. Leveraging pretrained Vision Transformers achieved **>90% accuracy**, demonstrating the absolute superiority of attention-based architectures for this benchmark.

## Tech Stack
- **Frameworks**: TensorFlow / Keras, Hugging Face, Scikit-Learn
- **Languages**: Python
