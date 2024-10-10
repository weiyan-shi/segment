from PIL import Image, ImageDraw
import os
import cv2

# 加载和预处理输入图像
image_path = "/app/Desktop/segment/pcit1/frames/2/image_with_boxes.jpg"
cur_img = cv2.imread(image_path)

# 创建输出文件夹
output_dir = "/app/Desktop/segment/pcit1/frames/2"
os.makedirs(output_dir, exist_ok=True)

gaze_data = {
    "person_0": {
        "gaze": [
            -0.6091488599777222,
            -0.7926972508430481,
            -0.02385082095861435
        ],
        "head_bbox": [
            624,
            188,
            821,
            446
        ]
    },
    "person_1": {
        "gaze": [
            0.680778443813324,
            -0.7108273506164551,
            -0.17681963741779327
        ],
        "head_bbox": [
            1150,
            426,
            1327,
            662
        ]
    }
}


# 遍历 gaze 数据
for person, data in gaze_data.items():
    gaze = data['gaze']
    head_bboxes = data['head_bbox']
    for xy in head_bboxes:
        xy = int(xy)
        head_center = [int(head_bboxes[1]+head_bboxes[3])//2,int(head_bboxes[0]+head_bboxes[2])//2]
        l = int(max(head_bboxes[3]-head_bboxes[1],head_bboxes[2]-head_bboxes[0])*1)
        gaze_len = 1000*1.0
        thick = max(5,int(l*0.01))
        cv2.arrowedLine(cur_img,(head_center[1],head_center[0]),
            (int(head_center[1]-gaze_len*gaze[0]),int(head_center[0]-gaze_len*gaze[1])),
            (230,253,11),thickness=thick)

# 保存修改后的图像
cv2.imwrite(os.path.join(output_dir, "image_with_gaze_vectors.jpg"), cur_img)

print(f"图像已保存到 {output_dir}")
