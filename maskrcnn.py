import torch
import torchvision
from PIL import Image, ImageDraw
import torchvision.transforms as T
import json
import os
import numpy as np

BASE_DIR = os.getenv('BASE_DIR', '/app/Desktop/Dataset/pcit1')
VIDEO_NAME = os.path.basename(BASE_DIR)

# COCO 数据集类别标签
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# 为不同类别分配颜色
CATEGORY_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 0),
    (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128), (64, 0, 0), (0, 64, 0),
    (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64), (192, 0, 0), (0, 192, 0), (0, 0, 192)
    # 可以根据需要添加更多颜色
]

# 加载预训练的 Mask R-CNN 模型
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 预处理变换
transform = T.Compose([T.ToTensor()])

# 创建总输出字典
all_output_data = {}

# 获取frames文件夹路径
frames_dir = os.path.join(BASE_DIR, 'frames')

# 检查frames文件夹是否存在
if not os.path.exists(frames_dir):
    raise ValueError(f"Frames folder not found at {frames_dir}")

# 获取frames文件夹中的所有图像文件
frame_files = [f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))]

# 创建分割结果保存目录
segment_dir = os.path.join(BASE_DIR, 'segment')
os.makedirs(segment_dir, exist_ok=True)

# 遍历每一帧图像
for frame_file in frame_files:
    image_path = os.path.join(frames_dir, frame_file)
    
    # 加载和预处理输入图像
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # 增加批次维度
    
    # 进行物体检测和分割
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # 初始化一个字典用于存储当前帧的边界框和其他信息
    frame_data = {}

    # 获取原始图像大小
    image_width, image_height = image.size

    # 转换 PIL 图像为 NumPy 数组以便处理掩码
    np_image = np.array(image)

    # 遍历检测结果
    for i, box in enumerate(predictions[0]['boxes']):
        score = predictions[0]['scores'][i].item()
        if score > 0.5:  # 设置置信度阈值
            # 提取边界框值
            xmin, ymin, xmax, ymax = box.int().tolist()

            # 获取类别标签编号并转换为类别名称
            label_idx = predictions[0]['labels'][i].item() - 1  # COCO 类别从1开始，而索引从0开始
            label_name = COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else "unknown"

            # 提取掩码并转换为二值化掩码
            mask = predictions[0]['masks'][i, 0].mul(255).byte().cpu().numpy()
            mask = Image.fromarray(mask)

            # 将掩码调整到原始图像大小并应用不同颜色
            mask_resized = mask.resize((image_width, image_height), resample=Image.BILINEAR)
            mask_np = np.array(mask_resized) > 128  # 将掩码转换为二值
            colored_mask = np.zeros_like(np_image, dtype=np.uint8)

            # 根据类别选择颜色
            color = CATEGORY_COLORS[label_idx % len(CATEGORY_COLORS)]
            colored_mask[mask_np] = color

            # 将掩码叠加到原始图像上
            np_image[mask_np] = np_image[mask_np] * 0.5 + colored_mask[mask_np] * 0.5

            # 将边界框和类别信息存储到当前帧的 JSON 中
            frame_data[f"object_{i}"] = {
                "box": {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                },
                "score": score,
                "label": label_idx + 1,  # 类别编号
                "label_name": label_name  # 类别名称
            }

    # 将 NumPy 数组转换回 PIL 图像
    segmented_image = Image.fromarray(np_image)

    # 保存分割结果图像
    segment_output_path = os.path.join(segment_dir, f"segmented_{frame_file}")
    segmented_image.save(segment_output_path)

    # 将当前帧的检测结果添加到总输出字典中
    all_output_data[frame_file] = frame_data

# 保存所有帧的边界框信息到一个 JSON 文件中
output_json_path = os.path.join(segment_dir, "all_frames_output_data.json")
with open(output_json_path, "w") as json_file:
    json.dump(all_output_data, json_file, indent=4)

print(f"所有帧的分割图像已保存到 {segment_dir}，边界框和类别信息已保存到 {output_json_path} 文件中。")
