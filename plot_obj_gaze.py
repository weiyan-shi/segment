import torch
import torchvision
from PIL import Image
import torchvision.transforms as T
import json
import os
import numpy as np
import cv2
import random


BASE_DIR = os.getenv('BASE_DIR', '/app/Desktop/Dataset/4')
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

# 生成颜色的函数，确保颜色的多样性
def generate_unique_colors(num_colors):
    random.seed(42)  # 保持结果可复现
    colors = []
    for _ in range(num_colors):
        color = [random.randint(0, 255) for _ in range(3)]
        colors.append(tuple(color))
    return colors

# 定义线段与矩形相交的函数
def line_intersects_bbox(start_point, end_point, bbox):

    xmin, ymin, xmax, ymax = bbox

    # 检查线段是否与 bbox 的四条边相交
    width = xmax - xmin
    height = ymax - ymin

    # 得到 rect: (左上角坐标 x, 左上角坐标 y, 宽度, 高度)
    rect = (xmin, ymin, width, height)

    intersects, pt1, pt2 = cv2.clipLine(rect, start_point, end_point)
    # print(intersects)

    return intersects

# 为 COCO 数据集每个类别生成颜色
CATEGORY_COLORS = generate_unique_colors(len(COCO_CLASSES))

# 加载预训练的 Mask R-CNN 模型
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 预处理变换
transform = T.Compose([T.ToTensor()])

# 加载 gaze 数据
gaze_json_path = os.path.join(BASE_DIR, f'{VIDEO_NAME}-gaze.json')
with open(gaze_json_path, 'r') as f:
    gaze_data = json.load(f)

# 创建分割结果保存目录
segment_dir = os.path.join(BASE_DIR, 'segment')
os.makedirs(segment_dir, exist_ok=True)

# 获取 frames 文件夹路径
frames_dir = os.path.join(BASE_DIR, 'frames')
if not os.path.exists(frames_dir):
    raise ValueError(f"Frames folder not found at {frames_dir}")

# 获取所有帧图像文件
frame_files = [f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))]

# 遍历每一帧图像
for frame_idx, frame_file in enumerate(frame_files):
    image_path = os.path.join(frames_dir, frame_file)
    frame_number = os.path.splitext(frame_file)[0]  # 去掉扩展名，得到 "2"

    # 加载图像并预处理
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # 增加批次维度
    
    # 获取原始图像尺寸
    image_width, image_height = image.size
    np_image = np.array(image)

    # 进行物体检测和分割
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # 获取当前帧的 gaze 信息
    gaze_info = gaze_data[frame_number]
    print(frame_number)
    if gaze_info is None:
        continue  # 如果没有 gaze 数据，则跳过该帧

    # 遍历该帧中的每个个体（person_0, person_1等）
    for person_id, data in gaze_info.items():
        gaze = data['gaze']
        head_bboxes = data['head_bbox']

        # 计算头部中心点
        head_center = [
            int((head_bboxes[0] + head_bboxes[2]) // 2),   # X 坐标
            int((head_bboxes[1] + head_bboxes[3]) // 2),  # Y 坐标
        ]
        
        # 计算视线的终点
        gaze_len = 1000 * 1.0  # 视线长度，可调
        end_point = (
            int(head_center[0] - gaze_len * gaze[0]),  # 计算 X 方向的终点
            int(head_center[1] - gaze_len * gaze[1])   # 计算 Y 方向的终点
        )

        # 绘制视线
        cv2.arrowedLine(np_image, (head_center[0], head_center[1]), end_point, (230, 253, 11), thickness=10)

        # 遍历物体检测结果
        for i, box in enumerate(predictions[0]['boxes']):
            score = predictions[0]['scores'][i].item()
            if score > 0.5:  # 置信度阈值
                xmin, ymin, xmax, ymax = box.int().tolist()
                bbox = (xmin, ymin, xmax, ymax)

                # 判断视线是否与物体边界框相交
                if line_intersects_bbox((head_center[0], head_center[1]), end_point, bbox):
                    # 匹配成功，画出物体的 Mask
                    mask = predictions[0]['masks'][i, 0].mul(255).byte().cpu().numpy()
                    mask_resized = Image.fromarray(mask).resize((image_width, image_height), resample=Image.BILINEAR)
                    mask_np = np.array(mask_resized) > 128  # 二值化

                    # 生成颜色的 Mask
                    colored_mask = np.zeros_like(np_image, dtype=np.uint8)
                    label_idx = predictions[0]['labels'][i].item() - 1
                    color = CATEGORY_COLORS[label_idx % len(CATEGORY_COLORS)]
                    colored_mask[mask_np] = color

                    # 叠加 Mask 到原图
                    np_image[mask_np] = np_image[mask_np] * 0.5 + colored_mask[mask_np] * 0.5

                    cv2.rectangle(np_image, (xmin, ymin), (xmax, ymax), color, thickness=2)
                    label_index = predictions[0]['labels'][i].item() - 1  # COCO class indices are 1-based, so subtract 1 for 0-based indexing
                    if label_index >= 0 and label_index < len(COCO_CLASSES):  # Ensure the index is valid
                        label = COCO_CLASSES[label_index]
                    else:
                        label = "Unknown"  # Handle cases where the label index is out of bounds
                    print(label)
                    cv2.putText(np_image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 保存带有视线和掩码的结果图像
    segmented_image = Image.fromarray(np_image)
    segment_output_path = os.path.join(segment_dir, f"{frame_file}")
    segmented_image.save(segment_output_path)

print(f"所有帧的分割图像已保存到 {segment_dir} 文件夹中。")
