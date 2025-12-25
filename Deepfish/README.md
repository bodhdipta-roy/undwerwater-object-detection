# DeepFish2 Notebook - Fish Detection & Analysis

## Overview
This Jupyter notebook implements three deep learning tasks for underwater fish analysis using the DeepFish dataset:

1. **Classification**: Binary classification for fish presence/absence detection
2. **Localization**: Fish counting using point annotations (⚠️ Currently not working)
3. **Segmentation**: Pixel-wise fish segmentation

## Dataset Structure

The notebook expects the DeepFish dataset in the following structure:

```
DeepFish/
├── Classification/
│   ├── train.csv
│   ├── val.csv
│   └── {habitat_code}/
│       ├── valid/
│       │   └── *.jpg
│       └── empty/
│           └── *.jpg
├── Localization/
│   ├── train.csv
│   ├── val.csv
│   ├── images/
│   │   └── *.jpg
│   └── masks/
│       └── *.png
└── Segmentation/
    ├── train.csv
    ├── val.csv
    ├── images/
    │   └── *.jpg
    └── masks/
        └── *.png
```

### Dataset Statistics
- **Classification**: 15,906 training images, 3,976 validation images
- **Localization**: 1,600 training images, 640 validation images
- **Segmentation**: 310 training images, 124 validation images

## Model Architectures

### 1. Classification Model (`FishClassifier`)
- **Backbone**: ResNet18 (pretrained on ImageNet)
- **Task**: Binary classification (fish present=1, fish absent=0)
- **Architecture**: ResNet18 with modified final layer (2 classes)
- **Parameters**: ~11 million

### 2. Localization Model (`FishLocalizer`)
- **Backbone**: FCN-ResNet50
- **Task**: Predict density maps for fish counting
- **Output**: Single-channel density map
- **Architecture**: Fully Convolutional Network with ResNet50 backbone

### 3. Segmentation Model (`FishSegmenter`)
- **Backbone**: FCN-ResNet50
- **Task**: Binary pixel-wise segmentation
- **Output**: Single-channel binary mask
- **Architecture**: Fully Convolutional Network with ResNet50 backbone

## Training Configuration

```python
BATCH_SIZE_CLF = 32  # Classification
BATCH_SIZE_LOC = 2   # Localization (memory constraints)
BATCH_SIZE_SEG = 2   # Segmentation (memory constraints)
NUM_WORKERS = 1
EPOCHS = 50
DEVICE = 'cuda' if available else 'cpu'
```

### Data Augmentation
**Training transforms**:
- Resize to 512×512
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.3)
- Color jitter (brightness, contrast, saturation ±0.2)
- Normalization (ImageNet stats)

**Validation transforms**:
- Resize to 512×512
- Normalization only

## Outputs

### Task 1: Classification ✅
**Model file**: `fish_classifier.pth`

**Training output**:
- Training and validation loss curves
- Training and validation accuracy curves
- Best validation accuracy saved
- Typical accuracy: ~90-95% (depending on training)

**Metrics tracked**:
- Training loss
- Training accuracy
- Validation loss
- Validation accuracy

### Task 2: Localization ⚠️ **NOT WORKING**
**Model file**: `fish_localizer.pth` (if training completes)

**Issues encountered**:
1. **Memory errors**: Training crashes due to GPU memory limitations
2. **Loss computation**: Tensor shape mismatches between predictions and ground truth
3. **Dataset size**: Large density maps require significant memory

**Error patterns**:
- `RuntimeError: CUDA out of memory`
- Shape mismatches during loss calculation
- Training interruptions (KeyboardInterrupt shown in notebook)

**What should work** (when fixed):
- Predict density maps for fish counting
- Output single-channel heatmaps
- MSE loss between predicted and ground truth density maps

### Task 3: Segmentation ✅
**Model file**: `fish_segmenter.pth`

**Training output**:
- Training and validation loss curves
- Binary segmentation masks
- BCEWithLogitsLoss for binary classification per pixel

**Metrics tracked**:
- Training loss
- Validation loss

### Additional Outputs
- `training_history.pkl`: Pickled dictionary containing training histories for classification and segmentation

## Problems Faced

### 1. Classification Dataset Path Issue (FIXED ✅)
**Problem**: Initial code assumed images were in `Classification/images/` subdirectory, but they're actually in `Classification/{habitat}/{valid|empty}/`

**Solution**: Modified `DeepFishClassification` dataset class to use correct path structure:
```python
# Images are directly in Classification folder, NOT in images subfolder!
self.images_base_dir = os.path.join(datadir, 'Classification')
```

### 2. Localization Training Failure (ONGOING ⚠️)
**Problem**: Training crashes with various errors

**Potential causes**:
- **Memory issues**: FCN-ResNet50 + large batch sizes exceed GPU memory
- **Density map generation**: Masks may not be proper density maps
- **Loss computation**: Shape mismatches between predictions and targets

**Attempted solutions**:
- Reduced batch size to 2
- Added dynamic shape resizing in training loop
- Memory cleanup with `gc.collect()` and `torch.cuda.empty_cache()`

**Still needs**:
- Proper density map generation from point annotations
- Memory-efficient training strategy
- Potentially lighter backbone (ResNet18 instead of ResNet50)

### 3. Memory Constraints
**Problem**: GPU memory exhaustion during training

**Solutions implemented**:
- Reduced batch sizes (2 for localization/segmentation)
- Used ResNet18 instead of ResNet50 for classification
- Pin memory disabled in some cases
- Garbage collection between tasks

### 4. Tensor Shape Mismatches
**Problem**: Output dimensions don't match target dimensions

**Solution**: Added interpolation in training loops:
```python
if outputs.shape[-2:] != masks.shape[-2:]:
    outputs = torch.nn.functional.interpolate(
        outputs, size=masks.shape[-2:], 
        mode='bilinear', align_corners=False
    )
```

## Hardware Requirements

**Minimum**:
- GPU: 4GB VRAM (GTX 1650 or better)
- RAM: 8GB system memory
- Storage: ~5GB for dataset + models

**Tested on**:
- GPU: GTX 1650 Super
- Python: 3.12.12
- PyTorch: Latest (CUDA-enabled)

## Usage

1. **Setup environment**:
```bash
pip install torch torchvision pandas pillow opencv-python tqdm matplotlib scikit-learn
```

2. **Update dataset path**:
```python
DATADIR = '/path/to/DeepFish'
```

3. **Run cells sequentially**:
   - Import libraries
   - Define dataset classes
   - Define models
   - Train classification ✅
   - (Skip localization ⚠️)
   - Train segmentation ✅

## Future Work

To fix localization:
1. **Generate proper density maps** from point annotations using Gaussian kernels
2. **Reduce model size**: Use ResNet18 instead of ResNet50
3. **Implement gradient checkpointing** for memory efficiency
4. **Try alternative counting approaches**: Detection-based counting instead of density maps
5. **Debug shape mismatches** systematically with detailed logging

## Files Generated

- `fish_classifier.pth` - Classification model weights
- `fish_localizer.pth` - Localization model weights (if training succeeds)
- `fish_segmenter.pth` - Segmentation model weights
- `training_history.pkl` - Training metrics and losses

## Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy
pandas
pillow
opencv-python
tqdm
matplotlib
scikit-learn
```

## References

- DeepFish Dataset: [Include dataset paper/source]
- ResNet: He et al., "Deep Residual Learning for Image Recognition"
- FCN: Long et al., "Fully Convolutional Networks for Semantic Segmentation"