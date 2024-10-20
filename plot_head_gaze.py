import torch
import torchvision
from PIL import Image
import torchvision.transforms as T
import json
import os
import numpy as np
import cv2
import random

BASE_DIR = os.getenv('BASE_DIR', '/app/Desktop/Dataset/3')
VIDEO_NAME = os.path.basename(BASE_DIR)

# 定义线段与矩形相交的函数
def line_intersects_bbox(start_point, end_point, bbox):
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    rect = (xmin, ymin, width, height)
    intersects, pt1, pt2 = cv2.clipLine(rect, start_point, end_point)
    return intersects

# 加载 gaze 数据
gaze_json_path = os.path.join(BASE_DIR, f'{VIDEO_NAME}-gaze.json')
with open(gaze_json_path, 'r') as f:
    gaze_data = json.load(f)

# 从视频中获取帧率
video_path = os.path.join(BASE_DIR, f'{VIDEO_NAME}.mp4')
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# 创建分割结果保存目录
segment_dir = os.path.join(BASE_DIR, 'onlyhead-segment')
os.makedirs(segment_dir, exist_ok=True)

# 记录注视事件的结果
gaze_events = []

# 获取 frames 文件夹路径
frames_dir = os.path.join(BASE_DIR, 'frames')
if not os.path.exists(frames_dir):
    raise ValueError(f"Frames folder not found at {frames_dir}")

# 获取所有帧图像文件
frame_files = [f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))]

# 遍历每一帧图像
for frame_idx, frame_file in enumerate(frame_files):
    try:
        image_path = os.path.join(frames_dir, frame_file)
        frame_number = os.path.splitext(frame_file)[0]  # 去掉扩展名，得到 "2"

        # 计算时间戳
        frame_time = int(frame_number) / fps

        # 加载图像并预处理
        image = Image.open(image_path).convert("RGB")
        np_image = np.array(image)

        # 获取当前帧的 gaze 信息
        gaze_info = gaze_data.get(frame_number, None)
        print(frame_number)
        if gaze_info is None:
            continue  # 如果没有 gaze 数据，则跳过该帧

        persons = list(gaze_info.keys())  # 获取所有人

        person0_looking_at_person1 = False
        person1_looking_at_person0 = False

        # 遍历该帧中的每个个体（person_0, person_1等）
        for person_id, data in gaze_info.items():
            try:
                gaze = data['gaze']
                head_bbox = data['head_bbox']

                # 计算头部中心点
                head_center = [
                    int((head_bbox[0] + head_bbox[2]) // 2),   # X 坐标
                    int((head_bbox[1] + head_bbox[3]) // 2),  # Y 坐标
                ]

                # 计算视线的终点
                gaze_len = 10000 * 1.0  # 视线长度，可调
                end_point = (
                    int(head_center[0] - gaze_len * gaze[0]),  # 计算 X 方向的终点
                    int(head_center[1] - gaze_len * gaze[1])   # 计算 Y 方向的终点
                )

                # 绘制视线
                cv2.arrowedLine(np_image, (head_center[0], head_center[1]), end_point, (230, 253, 11), thickness=10)

                # 检查该人的视线是否打在其他人的头部
                for other_person_id, other_data in gaze_info.items():
                    if person_id == other_person_id:
                        continue  # 跳过自身

                    other_head_bbox = other_data['head_bbox']

                    # 检查视线是否与其他人的头部相交
                    if line_intersects_bbox((head_center[0], head_center[1]), end_point, other_head_bbox):
                        # 绘制其他人头部的矩形框
                        cv2.rectangle(np_image, 
                                      (other_head_bbox[0], other_head_bbox[1]), 
                                      (other_head_bbox[2], other_head_bbox[3]), 
                                      (0, 0, 255), thickness=5)  # 用红色标出矩形框
                        print(f"{person_id} 的视线打在了 {other_person_id} 的头上！")

                        # 检查是否 person_0 看 person_1 或 person_1 看 person_0
                        if person_id == 'person_0' and other_person_id == 'person_1':
                            person0_looking_at_person1 = True
                        elif person_id == 'person_1' and other_person_id == 'person_0':
                            person1_looking_at_person0 = True

            except Exception as e:
                print(f"Error processing person {person_id} in frame {frame_number}: {e}")

        # 检查是否有相互注视事件
        if person0_looking_at_person1 and person1_looking_at_person0:
            gaze_events.append(frame_time)


        # 保存修改后的图像
        output_path = os.path.join(segment_dir, frame_file)
        cv2.imwrite(output_path, cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR))
    
    except Exception as e:
        print(f"Error processing frame {frame_file}: {e}")

gaze_events = sorted(gaze_events)

# 保存注视事件到JSON文件
gaze_events_json_path = os.path.join(BASE_DIR, f'{VIDEO_NAME}_gaze_events.json')
with open(gaze_events_json_path, 'w') as json_file:
    json.dump(gaze_events, json_file, indent=4)

print(f"所有帧的分割图像已保存到 {segment_dir} 文件夹中。")
print(f"注视事件已保存到 {gaze_events_json_path} 文件中。")
