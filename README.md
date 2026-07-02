# SAM2-Based Image Object Segmentation and Evaluation Framework

This project is built upon the **SAM2 (Segment Anything Model 2)** architecture and is designed for image object segmentation and quantitative evaluation. The framework supports reading image data and bounding box prompts, performing inference using SAM2, generating predicted segmentation masks, and computing a range of evaluation metrics, including IoU, Dice coefficient, MAE, MSE, Precision, Recall, F1-score, and FPS.

## Project Overview

This repository integrates the official SAM2 implementation and extends it with additional modules, including **AG-LoRA**, **Adapter-based tuning**, and **attention enhancement mechanisms**. It supports model inference, visualisation, and experimental evaluation for image segmentation tasks.

The primary entry point for inference and evaluation is:

```bash
val_f_s.py
```

This script processes a dataset based on JSON-formatted image names and bounding box prompts, performs segmentation inference on each image, and stores the resulting predictions in the specified output directory.

## Key Features

* Support for SAM2 / SAM2.1 model loading and inference
* Bounding box–guided instance segmentation
* Automatic mask generation
* Integration of LoRA, Adapter, and related parameter-efficient fine-tuning modules
* Batch processing of evaluation datasets
* Visualisation and saving of prediction results
* Comprehensive evaluation metrics:

  * Intersection over Union (IoU)
  * Dice Coefficient
  * Mean Absolute Error (MAE)
  * Mean Squared Error (MSE)
  * Precision
  * Recall
  * F1-score
  * Frames Per Second (FPS)

## Repository Structure

```bash
yjh/
├── sam2/
│   ├── configs/                    # SAM2 / SAM2.1 configuration files
│   ├── csrc/                       # CUDA / C++ extension modules
│   ├── modeling/                   # Core SAM2 model architecture
│   ├── utils/                      # Utility functions
│   ├── build_sam.py                # SAM2 model construction script
│   ├── sam2_image_predictor.py     # Image segmentation predictor
│   ├── automatic_mask_generator.py # Automatic mask generation module
│   ├── lora_sam2.py                # LoRA-based adaptation module
│   ├── lora_adapter.py             # Adapter-based tuning module
│   ├── Qlora.py                    # QLoRA implementation
│   └── ...
├── val_f_s.py                      # Main inference and evaluation script
├── .gitignore
└── README.md
```

## Environment Requirements

The framework is recommended to be executed in a Linux environment with CUDA support. CPU and Apple MPS backends are also supported; however, inference speed and numerical stability may be reduced.

Recommended configuration:

* Python ≥ 3.10
* PyTorch ≥ 2.0
* TorchVision
* OpenCV
* NumPy
* Matplotlib
* Pillow
* tqdm
* Hydra-Core
* OmegaConf

## Installation

Clone the repository:

```bash
git clone https://github.com/abiua/yjh.git
cd yjh
```

Install dependencies:

```bash
pip install torch torchvision
pip install opencv-python numpy matplotlib pillow tqdm hydra-core omegaconf
```

For CUDA-enabled environments, install the appropriate PyTorch version corresponding to your CUDA toolkit.

## Model Weights

Pre-trained SAM2 / SAM2.1 weights should be placed in the `checkpoints/` directory, for example:

```bash
checkpoints/checkpoint250pic-400.pt
```

The default checkpoint path in the code is:

```python
sam2_checkpoint = "./checkpoints/checkpoint250pic-400.pt"
```

If a different filename or directory structure is used, the path must be updated accordingly in `val_f_s.py`.

## Configuration Files

The default configuration is specified as:

```python
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
```

Alternative configurations may be selected depending on model scale:

```python
configs/sam2/sam2_hiera_t.yaml
configs/sam2/sam2_hiera_s.yaml
configs/sam2/sam2_hiera_b+.yaml
configs/sam2/sam2_hiera_l.yaml
```

Additional configuration files may be found under the `sam2/configs/` directory.

## Data Preparation

Prior to execution, the following components must be prepared:

### 1. Input Image Directory

A directory containing test images, for example:

```bash
testdata_orig/
```

### 2. Prompt Annotation File (Bounding Boxes)

A JSON file specifying bounding box prompts for each image:

```python
florence_prompt = "/path/to/labels_px_-m-renamed.json"
```

Example format:

```json
[
  {
    "file_name": "example.jpg",
    "bboxes": [
      [100, 120, 300, 360],
      [420, 180, 560, 400]
    ]
  }
]
```

### 3. Ground Truth Annotations

Ground truth data used for evaluation:

```python
ground_truth = "/path/to/val_coco_1.json"
```

COCO-style format example:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "example.jpg"
    }
  ],
  "annotations": [
    {
      "image_id": 1,
      "bbox": [100, 120, 200, 240]
    }
  ]
}
```

## Configuration of Execution Paths

Before running inference, update the following variables in `val_f_s.py`:

```python
florence_prompt = "/path/to/your/prompt.json"
input_folder = "/path/to/your/images"
output_folder = "/path/to/save/results"
ground_truth = "/path/to/your/ground_truth.json"
```

Example configuration:

```python
florence_prompt = "./data/prompts.json"
input_folder = "./data/images"
output_folder = "./output/results"
ground_truth = "./data/val_coco.json"
```

## Running Inference and Evaluation

After configuration, execute:

```bash
python val_f_s.py
```

The pipeline performs the following steps:

1. Load SAM2 model and pre-trained weights
2. Parse input JSON annotations
3. Iterate over test images
4. Select segmentation strategy based on number of bounding boxes
5. Generate predicted segmentation masks
6. Extract predicted bounding boxes
7. Compute evaluation metrics
8. Save visualised results
9. Report aggregated performance metrics

## Output Format

Predicted results are saved in the directory specified by `output_folder`, with filenames such as:

```bash
pred_example.jpg
```

Per-image evaluation outputs include:

```bash
IoU: 0.8234
Dice: 0.9012
MAE: 1
MSE: 1.0000
Precision: 0.8750
Recall: 0.7778
F1-score: 0.8235
FPS: 12.45 frames per second
```

After processing the full dataset, aggregated metrics are reported:

* Average IoU
* Average Dice coefficient
* Average MAE
* Average MSE
* Average Precision
* Average Recall
* Average F1-score

## Notes and Considerations

1. Ensure that the model checkpoint path is correctly specified prior to execution.
2. Verify that image filenames correspond exactly to those specified in the JSON prompt file.
3. Confirm GPU availability if CUDA is enabled:

```python
import torch
print(torch.cuda.is_available())
```

4. In cases of limited GPU memory, consider:

   * Using a smaller model configuration
   * Reducing input image resolution
   * Decreasing the number of bounding boxes per image
   * Switching to a lightweight (tiny/small) model variant

5. Many file paths are defined as local absolute paths and should be adapted to the user’s computing environment.

## Frequently Encountered Issues

### Missing checkpoint file

Ensure that:

```python
sam2_checkpoint = "./checkpoints/checkpoint250pic-400.pt"
```

points to a valid file.

### Image loading failure

Verify that:

```python
input_folder = "/path/to/your/images"
```

and that filenames match those in the JSON file.

### CUDA unavailable

Check PyTorch CUDA support:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If `False`, reinstall a CUDA-compatible PyTorch build.

### JSON format errors

Ensure the presence of required fields:

* `file_name`
* `bboxes`

Ground truth files must include:

* `images`
* `annotations`
* `bbox`
* `image_id`

## Future Improvements

* Introduction of a `requirements.txt` file
* Command-line argument interface to replace hard-coded paths
* Training pipeline documentation
* Example datasets and annotations
* Model checkpoint download instructions
* Experimental result tables
* Enhanced visualisation examples

## Licence

This project is developed based on the SAM2 framework. Users are required to comply with the original SAM2 licence and associated model usage regulations when deploying or distributing this codebase.
