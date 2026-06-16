# CXR-Net: Attention-Based Multi-Domain X-ray Image Classification

## Overview

CXR-Net is a domain-specific deep learning framework designed for automated X-ray image classification across multiple anatomical regions. Unlike conventional approaches that rely heavily on ImageNet-pretrained backbones, CXR-Net is built specifically for medical radiography and aims to learn discriminative features directly from X-ray images.

The architecture consists of three main components:

1. **CBR Block** – a lightweight convolutional building block for hierarchical feature extraction.
2. **CXR-Net Backbone** – a custom CNN architecture employing progressive feature fusion through hierarchical concatenation.
3. **S3 (Self-Spatial Synergy) Attention Block** – a hybrid attention module combining self-attention and spatial attention to exploit both global and local contextual information.

The framework is designed to be adaptable across different X-ray domains and imaging conditions while maintaining architectural simplicity and interpretability.

---

## Datasets

The model is evaluated on three publicly available X-ray datasets representing different clinical scenarios:

### Chest X-ray Dataset

The Chest X-ray dataset contains frontal chest radiographs categorized into:

* Normal
* Bacterial Pneumonia
* Viral Pneumonia

This dataset presents several challenges:

* Class imbalance
* Low-contrast pathological regions
* High visual similarity between bacterial and viral pneumonia
* Large variability in disease manifestation

---

### Bone Fracture Multi-Region X-ray Dataset

This dataset includes radiographs collected from multiple anatomical regions, including:

* Femur
* Tibia
* Fibula
* Radius
* Ulna
* Hips
* Knees
* Lumbar spine

Images are categorized into:

* Fractured
* Non-Fractured

The dataset contains diverse fracture types and varying imaging conditions, making robust feature extraction particularly important.

---

### Shoulder X-ray Dataset

The shoulder dataset consists of radiographs grouped into four clinical categories:

* A1
* C1
* D1
* Others

Compared with the other datasets, this dataset contains fewer samples and exhibits substantial intra-class variability, making it useful for evaluating the model's adaptability and generalization.

---

# Image Preprocessing

Medical X-ray images often contain subtle intensity differences and low-contrast structures.

To improve feature extraction and preserve diagnostic information, the preprocessing pipeline includes:

* Contrast enhancement
* Data augmentation

  * Horizontal flipping
  * Vertical flipping
  * Rotation

Augmentations that could distort anatomical structures, such as aggressive geometric transformations or unrealistic intensity manipulations, are intentionally avoided.

---

# CBR Block

The fundamental building block of CXR-Net is the **CBR block**, consisting of:

```
Convolution → Batch Normalization → ReLU
```

### Motivation

The CBR block was designed to provide:

* Progressive extraction of low-level and high-level image features
* Stable gradient propagation during training
* Better regularization through batch normalization
* Efficient learning with minimal architectural complexity

Convolutional layers capture image patterns ranging from simple edges to complex anatomical structures, while batch normalization improves optimization stability and ReLU introduces non-linearity for learning complex representations.

Because of its modular design, multiple CBR blocks can be stacked to construct deeper architectures while maintaining training stability.

---

# CXR-Net Architecture

CXR-Net is a custom convolutional neural network specifically designed for X-ray image analysis.

Unlike traditional CNNs that employ purely sequential feature extraction or residual shortcuts, CXR-Net introduces **hierarchical feature concatenation**.

### Why hierarchical concatenation?

Medical X-ray images often contain:

* Subtle abnormalities
* Low-contrast lesions
* Fine structural details
* Large anatomical variations

Hierarchical concatenation allows the network to:

* Reuse features extracted at different depths
* Preserve fine-grained information
* Improve gradient flow
* Combine low-level and high-level representations

After every two CBR blocks, feature maps are concatenated along the channel dimension, enabling richer feature fusion throughout the network.

This strategy differs from conventional residual connections by explicitly preserving and reusing intermediate representations rather than learning additive corrections.

---

# S3 Attention Block

At the end of the feature extraction stage, CXR-Net incorporates the proposed **Self-Spatial Synergy (S3)** attention module.

The S3 block combines:

* **Self-Attention**
* **Spatial Attention**

to simultaneously model global and local dependencies.

---

## Why Self-Attention?

Self-attention enables the network to capture long-range dependencies between distant regions in the image.

This is particularly important for X-ray images because:

* Pathological patterns may extend across large anatomical regions.
* Relationships between distant structures may provide diagnostic information.
* Global context helps the model understand the overall image composition.

However, self-attention alone may distribute attention too broadly and overlook localized details.

---

## Why Spatial Attention?

Spatial attention emphasizes important local regions by learning where the model should focus.

It helps:

* Highlight suspicious anatomical regions
* Preserve local textures and edges
* Refine feature maps spatially
* Suppress irrelevant background information

Nevertheless, spatial attention alone cannot model long-range relationships effectively.

---

## Why Combine Them?

The S3 block was designed to exploit the complementary strengths of both mechanisms.

By integrating self-attention and spatial attention:

* Global contextual relationships are preserved.
* Important local regions are emphasized.
* The network adaptively focuses on the most informative features.
* Feature representations become more robust across different anatomical regions and imaging conditions.

This hybrid attention mechanism enables CXR-Net to learn both broad structural patterns and fine-grained anatomical details within X-ray images.

---


# Citation

If you find this repository useful, please cite the associated publication.
