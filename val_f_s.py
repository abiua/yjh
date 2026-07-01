import os
import time
import cv2
from matplotlib import patches

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import json
import torch
import torchvision
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm  # 导入tqdm库，用于显示进度条

# 显示PyTorch和TorchVision的版本
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

# 如果使用Apple MPS设备，设置回退为CPU
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 选择计算设备
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # 使用bfloat16进行整个推理过程
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # 开启tfloat32支持（仅对Ampere GPU有效）
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

np.random.seed(3)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # 导入自动分割生成器


sam2_checkpoint = "./checkpoints/checkpoint250pic-400.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)
mask_generator = SAM2AutomaticMaskGenerator(sam2_model)  # 初始化自动分割生

# 显示预测框
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_mask(mask, ax, random_color=False, alpha=0.6):
    """
    在 ax 上显示单个 mask 的半透明填充 + 白色边 + 同色粗框 + 中心小矩形标签。
    mask: [H, W] 二值或 0/1 浮点
    """
    # 随机或固定 RGBA 颜色
    if random_color:
        c = np.concatenate([np.random.random(3), [alpha]])
    else:
        c = np.array([30/255, 144/255, 255/255, alpha])
    # 半透明填充
    ax.imshow(mask, cmap='gray', alpha=0)  # 先不让它自己画
    h, w = mask.shape
    # 1) 填充
    ax.imshow(np.dstack([mask*col for col in c]), alpha=1)
    # 2) 二值化提取轮廓 & bounding box
    ys, xs = np.where(mask > 0.5)
    if len(xs)==0 or len(ys)==0:
        return
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    bw, bh = x1-x0, y1-y0

    # 白色细边
    ax.add_patch(patches.Rectangle(
        (x0, y0), bw, bh,
        linewidth=1, edgecolor='white', facecolor='none',
        linestyle='-'))

    # # 同色粗框
    # ax.add_patch(patches.Rectangle(
    #     (x0, y0), bw, bh,
    #     linewidth=2, edgecolor=c[:3], facecolor='none',
    #     linestyle='-'))

    # 中心小圆角背景 + 文字
    label = "fish"
    # 文字尺寸估算
    txt_w = 6 * len(label)   # 大概 6px/字符
    txt_h = 10               # 大概 10px 高
    pad = 4
    box_w = txt_w + 2*pad
    box_h = txt_h + 2*pad

    cx = x0 + bw/2
    cy = y0 + bh/2
    lx = cx - box_w/2
    ly = cy - box_h/2

    # 圆角矩形 (approx: FancyBbox)
    box = patches.FancyBboxPatch(
        (lx, ly), box_w, box_h,
        boxstyle="round,pad=0.3,rounding_size=4",
        linewidth=0,
        facecolor=c[:3],
        alpha=1.0)
    ax.add_patch(box)
    # 白色文字
    ax.text(
        lx+pad, ly+pad+txt_h*0.6,  # y 轴要往下移一点来对齐基线
        label, color='white',
        fontsize=8, weight='bold')

def show_mask1(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # print(mask_image)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        # mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='white', facecolor=(0, 0, 0, 0), lw=0))

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(19.20, 10.80))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        # if box_coords is not None:
        #     # boxes
        #     show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


def extract_bbox_from_mask(mask):
    """
    从掩码中提取外接矩形框（bounding box）。
    mask:  二值掩码，值为1表示目标区域，值为0表示背景。
    return: 边界框（[x, y, width, height]）
    """
    # 确保掩码是二维的，如果是三维，去掉第一个维度
    if mask.ndim == 3:
        # 去掉第一个维度，这样我们会得到一个2D数组，形状变为 [H, W]
        mask = mask.squeeze(axis=0)

    # 确保掩码是二值化的（0和1）
    mask = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 获取最大轮廓（假设是目标物体的轮廓）
        contour = max(contours, key=cv2.contourArea)
        # 计算外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        return [x, y, w, h]
    return [0, 0, 0, 0]  # 如果没有找到有效轮廓，返回一个空框

def extract_bboxes_from_masks(masks):
    """
    从一组掩码中提取每个掩码的外接矩形框。
    masks: 掩码列表，包含每个目标的掩码
    return: 边界框列表，格式为 [x, y, width, height]
    """
    bboxes = []
    for mask in masks:
        bbox = extract_bbox_from_mask(mask)
        bboxes.append(bbox)
    return bboxes


def calculate_mae(true_bboxes_count, predicted_bboxes_count):
    # MAE = 真实框数量 - 预测框数量 的绝对值
    mae = abs(true_bboxes_count - predicted_bboxes_count)
    return mae


def calculate_mse(true_count, predicted_count):
    """
    计算RMSE（均方根误差）来评估真实框数量与预测框数量的误差
    true_count: 真实框的数量
    predicted_count: 预测框的数量
    返回值：RMSE
    """
    # 计算误差
    error = true_count - predicted_count

    # 计算均方误差（MSE）
    mse = np.mean(error ** 2)
    return mse
def calculate_tp_fp_tn_fn(true_bboxes, predicted_bboxes, iou_threshold=0.5):
    tp = 0  # True Positive
    fp = 0  # False Positive
    fn = 0  # False Negative
    matched_true_bboxes = []  # 用于存储已匹配的真实框，避免重复计算

    # 计算 TP 和 FP
    for pred_bbox in predicted_bboxes:
        match_found = False
        for true_bbox in true_bboxes:
            iou = calculate_iou(true_bbox, pred_bbox)
            if iou > iou_threshold and true_bbox not in matched_true_bboxes:
                tp += 1  # 找到匹配的真实框，增加 TP
                matched_true_bboxes.append(true_bbox)  # 标记该真实框已匹配
                match_found = True
                break
        if not match_found:
            fp += 1  # 如果没有找到匹配的真实框，则为 FP

    # 计算 FN
    fn = len(true_bboxes) - len(matched_true_bboxes)  # 未匹配的真实框为 FN
    return tp, fp, fn

def calculate_precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0  # 精确率

def calculate_recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0  # 召回率

def calculate_f1_score(precision, recall):
    if precision + recall == 0:
        return 0  # 如果精确率和召回率之和为0，则F1-score为0
    return 2 * (precision * recall) / (precision + recall)  # F1-score

def get_true_bboxes_count_for_image(json_file, image_filename):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 获取 image_id 对应的真实框数量
    image_id = None
    for image in data['images']:
        if image['file_name'] == image_filename:
            image_id = image['id']
            break

    if image_id is None:
        return 0  # 如果找不到对应的图片，则返回0

    # 统计该图片的真实框数量
    true_bboxes_count = 0
    for annotation in data['annotations']:
        if annotation['image_id'] == image_id:
            true_bboxes_count += 1
    return true_bboxes_count

def get_true_bboxes_for_image(json_file, image_filename):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 获取 image_id 对应的图片
    image_id = None
    for image in data['images']:
        if image['file_name'] == image_filename:
            image_id = image['id']
            break

    if image_id is None:
        return []  # 如果找不到对应的图片，则返回空列表

    # 获取该图片的所有真实框
    true_bboxes = []
    for annotation in data['annotations']:
        if annotation.get('image_id') == image_id:
            true_bboxes.append(annotation['bbox'])  # 将每个真实框的坐标添加到列表中

    return true_bboxes

def calculate_iou(bbox1, bbox2):
    # 确保每个 bbox 是一个包含 4 个元素的列表
    if len(bbox1) != 4 or len(bbox2) != 4:
        raise ValueError("Each bounding box should contain exactly 4 elements: [x, y, width, height]")

    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # 计算交集
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    # 计算交集区域的宽度和高度
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection_area = inter_width * inter_height

    # 计算每个框的面积
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2

    # 计算IoU
    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def calculate_average_iou(true_bboxes, predicted_bboxes):
    iou_scores = []
    for pred_bbox in predicted_bboxes:
        best_iou = 0
        # 对每个预测框，计算它与所有真实框的IoU
        for true_bbox in true_bboxes:
            iou = calculate_iou(true_bbox, pred_bbox)
            best_iou = max(best_iou, iou)  # 选择与该预测框匹配度最高的真实框
        iou_scores.append(best_iou)

    # 计算平均IoU
    average_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0
    return average_iou

def calculate_dice(bbox1, bbox2):

    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # 计算交集
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    # 计算交集区域的宽度和高度
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection_area = inter_width * inter_height

    # 计算每个框的面积
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2

    # 计算Dice系数
    dice = (2 * intersection_area) / (bbox1_area + bbox2_area) if (bbox1_area + bbox2_area) > 0 else 0
    return dice


def calculate_average_dice(true_bboxes, predicted_bboxes):
    dice_scores = []
    for pred_bbox in predicted_bboxes:
        best_dice = 0
        # 对每个预测框，计算它与所有真实框的Dice系数
        for true_bbox in true_bboxes:
            dice = calculate_dice(true_bbox, pred_bbox)
            best_dice = max(best_dice, dice)  # 选择与该预测框匹配度最高的真实框
        dice_scores.append(best_dice)

    # 计算平均Dice系数
    average_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0
    return average_dice

# 初始化每个指标的列表，用于存储所有图片的结果
all_iou = []
all_dice = []
all_mae = []
all_mse = []
all_precision = []
all_recall = []
all_f1_score = []

def process_images_from_json(json_file_path, input_folder, output_folder, predictor):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 初始化计时器和处理的图像数量
    start_time = time.time()  # 记录开始时间
    processed_images = 0  # 计数已处理的图像

    # 使用tqdm显示处理进度，并实时显示当前处理的图片名称
    for entry in tqdm(data, desc="Processing images", dynamic_ncols=True):
        file_name = entry["file_name"]
        bboxes = np.array(entry["bboxes"])

        # 加载图像
        image_path = os.path.join(input_folder, file_name)
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        current_image_filename = file_name
        # 获取预测框的数量
        true_bboxes = get_true_bboxes_for_image(ground_truth, current_image_filename)
        predicted_bboxes = []

        # 推理阶段开始时间
        inference_start_time = time.time()

        # 根据框的数量选择不同的推理方法
        if len(bboxes) <= 1:
            # 框的数量小于或等于1时，使用自动分割
            print(f"Using weight method for {file_name} (No box or only one box)")
            masks = mask_generator.generate(image)
            for mask in masks:
                # 提取 'bbox' 字段
                if isinstance(mask, dict) and 'bbox' in mask:
                    bbox = mask['bbox']
                    predicted_bboxes.append(bbox)
            # 输出每张图片模型推理出来的mask个数
            print(f"Image: {file_name}, Predicted Masks Count: {len(masks)}")
            predicted_bboxes_count = len(masks)

            true_bboxes_count = get_true_bboxes_count_for_image(ground_truth, current_image_filename)
            mae = calculate_mae(true_bboxes_count, predicted_bboxes_count)
            mse = calculate_mse(true_bboxes_count, predicted_bboxes_count)
            tp, fp, fn = calculate_tp_fp_tn_fn(true_bboxes, predicted_bboxes, iou_threshold=0.5)
            precision = calculate_precision(tp, fp)
            recall = calculate_recall(tp, fn)
            f1_score = calculate_f1_score(precision, recall)
            average_iou = calculate_average_iou(true_bboxes, predicted_bboxes)
            average_dice = calculate_average_dice(true_bboxes, predicted_bboxes)
            print(f"IoU: {average_iou:.4f}")
            print(f"Dice: {average_dice:.4f}")
            print(f"MAE: {mae},MSE: {mse:.4f}")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")
            # 将结果添加到相应的列表中
            all_iou.append(average_iou)
            all_dice.append(average_dice)
            all_mae.append(mae)
            all_mse.append(mse)
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1_score.append(f1_score)
        else:
            # 框的数量大于1时，使用预测器进行推理
            print(f"Using predictor method for {file_name} (Multiple boxes)")
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bboxes,
                multimask_output=False,
            )
            for mask in masks:
                if isinstance(mask, np.ndarray):  # 检查掩码是否为数组类型
                    if mask.shape[0] == 1:  # 如果第一个维度大小为1，才执行squeeze
                        mask = mask.squeeze(0)
                        predicted_bboxes.append(mask)  # 保存预测框
            # 推理阶段结束时间
            inference_end_time = time.time()

            # 计算推理耗时和 FPS
            inference_elapsed_time = inference_end_time - inference_start_time
            inference_fps = 1 / inference_elapsed_time if inference_elapsed_time > 0 else 0  # 计算FPS

            # 输出每张图片模型推理出来的mask个数
            print(f"Image: {file_name}, Predicted Masks Count: {len(masks)}")
            print(f"FPS: {inference_fps:.2f} frames per second")
            predicted_bboxes_count = len(masks)
            true_bboxes_count = get_true_bboxes_count_for_image(ground_truth, current_image_filename)
            mae = calculate_mae(true_bboxes_count, predicted_bboxes_count)
            mse = calculate_mse(true_bboxes_count, predicted_bboxes_count)
            # 从掩码中提取预测框
            predicted_bboxes = extract_bboxes_from_masks(masks)
            print(f"Mean Absolute Error (MAE) for {current_image_filename}: {mae}")
            tp, fp, fn = calculate_tp_fp_tn_fn(true_bboxes, predicted_bboxes, iou_threshold=0.5)
            precision = calculate_precision(tp, fp)
            recall = calculate_recall(tp, fn)
            f1_score = calculate_f1_score(precision, recall)
            average_iou = calculate_average_iou(true_bboxes, predicted_bboxes)
            average_dice = calculate_average_dice(true_bboxes, predicted_bboxes)
            print(f"IoU: {average_iou:.4f}")
            print(f"Dice: {average_dice:.4f}")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")
            print(f"RMSE:{mse:.4f}")
            # 将结果添加到相应的列表中
            all_iou.append(average_iou)
            all_dice.append(average_dice)
            all_mae.append(mae)
            all_mse.append(mse)
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1_score.append(f1_score)
        # 可视化并保存预测结果
        fig, ax = plt.subplots(figsize=(19.20, 10.80))
        ax.imshow(image)
        ax.axis('off')  # 关闭坐标轴
        # 处理掩码并显示
        for mask in masks:
            if isinstance(mask, np.ndarray):  # 检查掩码是否为数组类型
                if mask.shape[0] == 1:  # 如果第一个维度大小为1，才执行squeeze
                    mask = mask.squeeze(0)
                # 计算掩码占比
                mask_area = np.sum(mask > 0)  # 掩码区域像素数量
                total_area = mask.size  # 图像总像素数量
                mask_ratio = mask_area / total_area  # 计算占比

                # 只有当掩码占比大于50%时才显示
                if mask_ratio < 0.5:
                    show_mask(mask, ax, random_color=True)

            elif isinstance(mask, dict):  # 如果掩码是字典类型，提取掩码
                mask_data = mask.get('segmentation', None)
                if mask_data is not None:
                    mask_data = mask_data.astype(np.float32)

                    # 计算掩码占比
                    mask_area = np.sum(mask_data > 0)  # 掩码区域像素数量
                    total_area = mask_data.size  # 图像总像素数量
                    mask_ratio = mask_area / total_area  # 计算占比

                    # 只有当掩码占比大于50%时才显示
                    if mask_ratio < 0.5:
                        show_mask1(mask_data, plt.gca(), random_color=True)
        ax.axis('off')
        output_image_path = os.path.join(output_folder, f"pred_{file_name}")
        plt.savefig(output_image_path, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"预测结果已保存到: {output_image_path}")
        processed_images += 1  # 更新处理的图像数量

    # 计算推理阶段的总体 FPS
    end_time = time.time()  # 记录结束时间
    total_inference_time = end_time - start_time  # 计算总耗时
    total_fps = processed_images / total_inference_time if total_inference_time > 0 else 0  # 计算整体FPS
    print(f"Processed {processed_images} images in {total_inference_time:.2f} seconds.")
    print(f"Total FPS: {total_fps:.2f} frames per second")


if __name__ == "__main__":
    # 文件路径
    # florence_prompt = '/home/ai/data/GGbond/nokk/SAM2/trainmask/imagemaskjson/box_c2tssa_baddream.json'  # JSON 文件路径
    # input_folder = '/home/ai/data/GGbond/nokk/SAM2/generalization1_1'  # 输入图像文件夹
    # output_folder = '/home/ai/data/GGbond/nokk/SAM2/output/yolo_c2tssa_baddream_pic'  # 输出结果文件夹
    # ground_truth = '/home/ai/data/GGbond/nokk/SAM2/generalization1_1/bad_dream.json'
    # florence_prompt = '/home/ai/data/GGbond/nokk/SAM2/trainmask/imagemaskjson/labels_px_-m-renamed.json'  # JSON 文件路径
    # input_folder = '/home/ai/data/GGbond/nokk/SAM2/testdata_orig'  # 输入图像文件夹
    # output_folder = '/home/ai/data/GGbond/nokk/SAM2/output/yolo_m'  # 输出结果文件夹
    # ground_truth = '/home/ai/data/GGbond/nokk/SAM2/trainmask/imagemaskjson/val_coco_1.json'
    florence_prompt = '/home/ai/data/GGbond/nokk/SAM2/trainmask/imagemaskjson/labels_px_-m-renamed.json'  # JSON 文件路径
    input_folder = '/home/ai/data/GGbond/nokk/SAM2/testdata_orig'  # 输入图像文件夹
    output_folder = '/home/ai/data/GGbond/nokk/SAM2/output/sam_lora_42'  # 输出结果文件夹
    ground_truth = '/home/ai/data/GGbond/nokk/SAM2/trainmask/imagemaskjson/val_coco_1.json'
    # 批量处理图像并保存预测结果
    process_images_from_json(florence_prompt, input_folder, output_folder, predictor)
    # 计算平均值
    average_iou_all = sum(all_iou) / len(all_iou) if all_iou else 0
    average_dice_all = sum(all_dice) / len(all_dice) if all_dice else 0
    average_mae_all = sum(all_mae) / len(all_mae) if all_mae else 0
    average_mse_all = sum(all_mse) / len(all_mse) if all_mse else 0
    average_precision_all = sum(all_precision) / len(all_precision) if all_precision else 0
    average_recall_all = sum(all_recall) / len(all_recall) if all_recall else 0
    average_f1_score_all = sum(all_f1_score) / len(all_f1_score) if all_f1_score else 0
    # 输出平均值
    print(f"Average IoU for all images: {average_iou_all:.4f}")
    print(f"Average Dice Coefficient for all images: {average_dice_all:.4f}")
    print(f"Average MAE for all images: {average_mae_all:.4f}")
    print(f"Average MSE for all images: {average_mse_all:.4f}")
    print(f"Average Precision for all images: {average_precision_all:.4f}")
    print(f"Average Recall for all images: {average_recall_all:.4f}")
    print(f"Average F1-score for all images: {average_f1_score_all:.4f}")
