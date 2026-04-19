# HW2 - Visual Recognition using Deep Learning (Digit Detection with DETR)
Titouan GAGNEUX, ID : 114550830

## Introduction

This project is part of the *Visual Recognition using Deep Learning* course (Spring 2026).

The objective of this homework is to solve a **digit detection problem**, where the model must both:
- **localize** digits in an image (bounding boxes),
- **classify** each detected digit.

Unlike a standard classification task, this problem follows the **object detection paradigm** and requires predicting multiple objects per image.

The dataset contains:

* 30,062 training images  
* 3,340 validation images  
* 13,068 test images  

Annotations are provided in **COCO format**, with bounding boxes defined as:
[x_min, y_min, width, height]


The goal is to achieve the highest possible performance on the competition leaderboard while respecting the constraints:

* Use **DETR (DEtection TRansformer)** as the model
* Use **ResNet-50 as backbone**
* No external data allowed

---

## Environment Setup

### Requirements

* Python >= 3.9  
* PyTorch  
* torchvision  
* transformers (Hugging Face)  
* NumPy  
* matplotlib  
* tqdm  

### Installation

```bash
pip install -r requirements.txt
```

---

## Method Overview
### Data Preprocessing

The dataset is loaded in COCO format and processed to ensure compatibility with DETR.

Key preprocessing steps:

* Filtering invalid bounding boxes (too small or out-of-bound)
* Conversion of category IDs to contiguous labels (0 to N-1)
* Image normalization using ImageNet statistics
* Automatic resizing and padding handled by DetrImageProcessor

Data augmentation techniques used during training:

* Random horizontal flipping
* Color jittering (brightness, contrast, saturation)

For validation:

* No augmentation is applied (only preprocessing)

---

## Model Architecture

The model is based on DETR (DEtection TRansformer).

It consists of:

* A ResNet-50 backbone for feature extraction
* A transformer encoder-decoder for global reasoning
* A fixed number of object queries for prediction

Two configurations were explored:

1. Initial approach (from scratch DETR)

The backbone (ResNet-50) was initialized with pretrained ImageNet weights, while:

* transformer encoder and decoder
* detection heads

were trained from scratch.

However, this configuration led to:

* slow convergence,
* high validation loss,
* unstable training behavior.

  
2. Final approach (pretrained DETR)

To obtain reliable results, a fully pretrained DETR model was used.

This approach allows:

* faster convergence,
* more stable training,
* significantly improved detection performance.

The final reported results are based on this pretrained DETR configuration.

---

## Training Strategy
* Loss function: DETR loss (classification + bbox + GIoU + auxiliary losses)
* Optimizer: AdamW
* Learning rates:
* Transformer: 1e-4
* Backbone: 1e-5
* Learning rate scheduler: Cosine Annealing
* Mixed precision training (AMP)
* Gradient clipping

Additional details:

* Batch training with DataLoader
* Validation at each epoch
* Best model saved based on validation loss

---

## Usage

Train the model

```bash
python train.py
```

Run inference (generate pred.json)

```bash
python inference.py
```

The output file follows COCO format:

```bash
{
  "image_id": int,
  "bbox": [x, y, w, h],
  "score": float,
  "category_id": int
}
```

---

## Performance Snapshot
Final results obtained using pretrained DETR : 0.3
Training / validation curves :

Leaderboard :
<img width="772" height="35" alt="image" src="https://github.com/user-attachments/assets/2bf0a67a-0c7c-47a0-8d97-ccd1fc2d3462" />

---

## Notes

A significant part of the work was dedicated to training DETR from scratch.

### However, this approach proved difficult due to:
* slow convergence,
* high computational cost,
* limited resources

### Practical constraints :
* One training run of 50 epochs required more than 16 hours.
* No local GPU was available.
* All experiments were conducted using Google Colab (paid version).
* Compute units are limited and must be repurchased, restricting experimentation.
* Frequent interruptions and restarts increased overall training time.

Due to these constraints, it was not possible to fully optimize the from-scratch DETR within the available time.

As a result, a pretrained DETR model was used for the final solution to ensure:

* stable training,
* correct predictions,
* competitive performance

--- 

## References
* DETR Paper: https://arxiv.org/abs/2005.12872
* PyTorch Documentation: https://pytorch.org/
* Hugging Face Transformers: https://huggingface.co/docs/transformers
* Torchvision Models: https://pytorch.org/vision/stable/models.html
* Course materials

