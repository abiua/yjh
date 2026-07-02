# yjh

基于 **SAM2** 的图像目标分割与评估项目。
本项目主要用于读取图像与检测框提示信息，调用 SAM2 模型进行目标分割，生成预测掩码结果，并计算 IoU、Dice、MAE、MSE、Precision、Recall、F1-score、FPS 等评估指标。

## 项目简介

本仓库集成了 SAM2 相关模型代码，并加入了 AG-LoRA、Adapter、Attention 等扩展模块，可用于图像分割模型的推理、可视化与实验评估。

当前主程序为：

```bash
val_f_s.py
```

该脚本会根据输入 JSON 中的图片名称和目标框信息，对测试图片进行分割预测，并将预测结果保存到指定输出目录。

## 主要功能

* 支持 SAM2 / SAM2.1 模型加载与推理
* 支持基于 bounding box 的目标分割
* 支持自动掩码生成
* 支持 LoRA / Adapter 等模型扩展结构
* 支持批量处理测试图片
* 支持预测结果可视化保存
* 支持常用分割与检测评估指标计算：

  * IoU
  * Dice Coefficient
  * MAE
  * MSE
  * Precision
  * Recall
  * F1-score
  * FPS

## 项目结构

```bash
yjh/
├── sam2/
│   ├── configs/                    # SAM2 / SAM2.1 配置文件
│   ├── csrc/                       # CUDA / C++ 扩展相关代码
│   ├── modeling/                   # SAM2 模型结构代码
│   ├── utils/                      # 工具函数
│   ├── build_sam.py                # 构建 SAM2 模型
│   ├── sam2_image_predictor.py     # 图像分割预测器
│   ├── automatic_mask_generator.py # 自动掩码生成器
│   ├── lora_sam2.py                # LoRA 相关模块
│   ├── lora_adapter.py             # Adapter 相关模块
│   ├── Qlora.py                    # QLoRA 相关模块
│   └── ...
├── val_f_s.py                      # 主推理与评估脚本
├── .gitignore
└── README.md
```

## 环境要求

建议使用支持 CUDA 的 Linux 环境运行。
如果没有 CUDA，也可以在 CPU 或 Apple MPS 上运行，但推理速度和结果稳定性可能会受到影响。

推荐环境：

```bash
Python >= 3.10
PyTorch >= 2.0
TorchVision
OpenCV
NumPy
Matplotlib
Pillow
tqdm
Hydra
OmegaConf
```

## 安装依赖

克隆仓库：

```bash
git clone https://github.com/abiua/yjh.git
cd yjh
```

安装常用依赖：

```bash
pip install torch torchvision
pip install opencv-python numpy matplotlib pillow tqdm hydra-core omegaconf
```

如果使用 CUDA，请根据自己的 CUDA 版本安装对应的 PyTorch 版本。

## 权重文件准备

请将训练好的 SAM2 / SAM2.1 权重文件放入 `checkpoints/` 目录，例如：

```bash
checkpoints/checkpoint250pic-400.pt
```

默认脚本中的权重路径为：

```python
sam2_checkpoint = "./checkpoints/checkpoint250pic-400.pt"
```

如果你的权重文件名称或路径不同，需要在 `val_f_s.py` 中修改对应路径。

## 配置文件

默认使用的模型配置文件为：

```python
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
```

如果需要切换模型规模或配置，可以修改为其他配置文件，例如：

```python
model_cfg = "configs/sam2/sam2_hiera_t.yaml"
model_cfg = "configs/sam2/sam2_hiera_s.yaml"
model_cfg = "configs/sam2/sam2_hiera_b+.yaml"
model_cfg = "configs/sam2/sam2_hiera_l.yaml"
```

或使用 `sam2/configs/` 目录下的其他训练配置文件。

## 数据准备

运行前需要准备以下内容：

### 1. 输入图片目录

存放待测试图片，例如：

```bash
testdata_orig/
```

### 2. 提示框 JSON 文件

用于提供每张图片对应的目标框信息。脚本中默认变量为：

```python
florence_prompt = "/path/to/labels_px_-m-renamed.json"
```

JSON 数据格式示例：

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

### 3. Ground Truth 标注文件

用于计算评估指标。脚本中默认变量为：

```python
ground_truth = "/path/to/val_coco_1.json"
```

Ground Truth 推荐使用 COCO 格式，例如：

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

## 修改运行路径

在运行前，请打开 `val_f_s.py`，根据自己的数据位置修改以下路径：

```python
florence_prompt = "/path/to/your/prompt.json"
input_folder = "/path/to/your/images"
output_folder = "/path/to/save/results"
ground_truth = "/path/to/your/ground_truth.json"
```

例如：

```python
florence_prompt = "./data/prompts.json"
input_folder = "./data/images"
output_folder = "./output/results"
ground_truth = "./data/val_coco.json"
```

## 运行推理与评估

修改路径后，直接运行：

```bash
python val_f_s.py
```

程序会自动完成以下流程：

1. 加载 SAM2 模型与权重
2. 读取输入 JSON
3. 逐张读取测试图片
4. 根据目标框数量选择分割方式
5. 生成预测 mask
6. 提取预测框
7. 计算评价指标
8. 保存可视化结果
9. 输出平均 IoU、Dice、MAE、MSE、Precision、Recall、F1-score 等指标

## 输出结果

预测结果会保存到 `output_folder` 指定目录中，文件名格式类似：

```bash
pred_example.jpg
```

终端会输出每张图片的指标信息，例如：

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

全部图片处理完成后，会输出整体平均指标：

```bash
Average IoU for all images
Average Dice Coefficient for all images
Average MAE for all images
Average MSE for all images
Average Precision for all images
Average Recall for all images
Average F1-score for all images
```

## 注意事项

1. 运行前请确认权重文件路径正确。
2. 运行前请确认输入图片文件名与 JSON 中的 `file_name` 完全一致。
3. 如果使用 CUDA，建议确认 PyTorch 能够正常识别 GPU：

```python
import torch
print(torch.cuda.is_available())
```

4. 如果显存不足，可以尝试：

   * 使用更小的模型配置
   * 减小输入图片尺寸
   * 减少单次处理的目标框数量
   * 切换到 tiny / small 版本模型

5. 当前脚本中的部分路径为本地绝对路径，需要根据自己的服务器或电脑目录进行修改。

## 常见问题

### 1. 找不到权重文件

请检查：

```python
sam2_checkpoint = "./checkpoints/checkpoint250pic-400.pt"
```

确保该路径下存在对应 `.pt` 文件。

### 2. 找不到图片

请检查：

```python
input_folder = "/path/to/your/images"
```

并确认 JSON 中的 `file_name` 与图片真实文件名一致。

### 3. CUDA 不可用

可以先检查 PyTorch 是否识别 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

如果返回 `False`，需要重新安装与 CUDA 版本匹配的 PyTorch。

### 4. JSON 格式报错

请确认输入 JSON 是合法格式，并且包含必要字段：

```json
file_name
bboxes
```

Ground Truth 文件需要包含：

```json
images
annotations
bbox
image_id
```

## 后续改进方向

* 增加 `requirements.txt`
* 增加命令行参数，避免手动修改脚本路径
* 增加训练脚本说明
* 增加示例数据格式
* 增加模型权重下载说明
* 增加实验结果表格
* 增加可视化示例图片

## License

本项目基于 SAM2 相关代码进行开发。
如使用本仓库代码，请遵守原始 SAM2 项目的开源协议及相关模型权重使用规定。

