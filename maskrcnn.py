import torch
import torchvision
from PIL import Image, ImageDraw
import torchvision.transforms as T
import json
import os

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

# 加载预训练的 Mask R-CNN 模型
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 加载和预处理输入图像
image = Image.open("/app/Desktop/Dataset/pcit2/frames/11.jpg")
transform = T.Compose([T.ToTensor()])
image_tensor = transform(image).unsqueeze(0)  # 增加批次维度

# 进行物体检测和分割
with torch.no_grad():
    predictions = model(image_tensor)

# 创建输出文件夹
output_dir = "pcit2/frames/11"
os.makedirs(output_dir, exist_ok=True)

# 初始化一个字典用于存储边界框和其他信息
output_data = {}

# 获取原始图像大小
image_width, image_height = image.size

# 在原始图像上绘制边界框
draw = ImageDraw.Draw(image)

# 遍历检测结果
for i, box in enumerate(predictions[0]['boxes']):
    score = predictions[0]['scores'][i].item()
    if score > 0.5:  # 设置置信度阈值
        # 提取边界框值
        xmin, ymin, xmax, ymax = box.int().tolist()

        # 在原始图像上绘制边界框
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

        # 获取类别标签编号并转换为类别名称
        label_idx = predictions[0]['labels'][i].item() - 1  # COCO 类别从1开始，而索引从0开始
        label_name = COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else "unknown"

        # 在图像上绘制类别名称和置信度
        draw.text((xmin, ymin), f"{label_name} {score:.2f}", fill="red")

        # 提取并保存对象图像（基于边界框切割）
        object_image = image.crop((xmin, ymin, xmax, ymax))
        object_filename = f"object_{i}.png"
        object_image.save(os.path.join(output_dir, object_filename))

        # 将边界框和类别信息存储到 JSON 中
        output_data[f"object_{i}"] = {
            "filename": object_filename,
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

# 保存带有边界框的原图
image.save(os.path.join(output_dir, "image_with_boxes.jpg"))

# 保存 JSON 文件
with open("output_data.json", "w") as json_file:
    json.dump(output_data, json_file, indent=4)

print(f"所有对象图片已保存到 {output_dir}，边界框和类别信息已保存到 output_data.json 文件中，带边框的原图已保存。")
