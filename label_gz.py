import os
import cv2
import pandas as pd
from tkinter import Tk, Button, Label, StringVar
from PIL import Image, ImageTk

def load_annotations():
    """载入或初始化标记文件"""
    global annotations, csv_path
    try:
        annotations = pd.read_csv(csv_path, index_col="frame")
    except FileNotFoundError:
        annotations = pd.DataFrame(columns=["frame", "child_label", "parent_label"]).set_index("frame")

def update_frame():
    """显示current frame并更新界面"""
    global frame_index, child_label, parent_label

    # 设置current frame位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        frame_text.set("video end")
        return

    # 从标注数据中载入已标注标签
    if frame_index in annotations.index:
        child_label = annotations.loc[frame_index, "child_label"] == "True"
        parent_label = annotations.loc[frame_index, "parent_label"] == "True"
    else:
        child_label, parent_label = None, None

    # print(frame_index, child_label, parent_label)

    # 将视频帧转换为 Tkinter 显示格式
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    frame_label.imgtk = imgtk
    frame_label.configure(image=imgtk)
    
    # 更新帧信息
    frame_text.set(f"current frame: {frame_index}")

    # 更新状态显示
    update_status_texts()
    # 更新按钮颜色
    update_button_colors()

    root.update_idletasks()  # 强制刷新

def update_status_texts():
    print(frame_index, child_label, parent_label)
    """更新标签状态显示"""
    if child_label is True:
        print(1)
        child_status_text.set("Child Label: True")
    elif child_label == "False" or child_label is False:
        child_status_text.set("Child Label: False")
    else:
        child_status_text.set("Child Label: None")

    if parent_label is True:
        parent_status_text.set("Parent Label: True")
    elif parent_label is False:
        parent_status_text.set("Parent Label: False")
    else:
        parent_status_text.set("Parent Label: None")


def update_button_colors():
    """更新按钮颜色来显示当前选择状态"""
    for btn in [btn_child_true, btn_child_false, btn_parent_true, btn_parent_false]:
        btn.configure(bg="lightgrey", activebackground="lightgrey")

    if child_label is True:
        btn_child_true.configure(bg="lightgreen", activebackground="lightgreen")
    elif child_label is False:
        btn_child_false.configure(bg="lightgreen", activebackground="lightgreen")

    if parent_label is True:
        btn_parent_true.configure(bg="lightblue", activebackground="lightblue")
    elif parent_label is False:
        btn_parent_false.configure(bg="lightblue", activebackground="lightblue")

def save_annotation():
    """保存标记到 DataFrame"""
    if child_label is not None and parent_label is not None:
        annotations.loc[frame_index] = [child_label, parent_label]
        annotations.to_csv(csv_path, index_label="frame")

def mark_child_true():
    global child_label
    child_label = True
    update_status_texts()  # 及时更新状态显示
    save_annotation()

def mark_child_false():
    global child_label
    child_label = False
    update_status_texts()  # 及时更新状态显示
    save_annotation()

def mark_parent_true():
    global parent_label
    parent_label = True
    update_status_texts()  # 及时更新状态显示
    save_annotation()

def mark_parent_false():
    global parent_label
    parent_label = False
    update_status_texts()  # 及时更新状态显示
    save_annotation()

def next_frame():
    """跳到下一个 8 帧"""
    global frame_index
    frame_index += 8
    update_frame()

def previous_frame():
    """返回上一个 8 帧"""
    global frame_index
    frame_index = max(0, frame_index - 8)
    update_frame()

def on_closing():
    cap.release()
    root.destroy()

def main():
    global cap, frame_index, child_label, parent_label
    global btn_child_true, btn_child_false, btn_parent_true, btn_parent_false
    global frame_label, frame_text, child_status_text, parent_status_text
    global annotations, csv_path, root

    # 视频和标注初始化
    video_path = "dataset/3-gaze.mp4"
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = f"{video_name}_annotations.csv"
    load_annotations()
    
    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    child_label, parent_label = None, None

    # Tkinter 窗口设置
    root = Tk()
    root.title("label tool")

    # 显示视频帧
    frame_label = Label(root)
    frame_label.grid(row=0, column=0, columnspan=7)

    # current frame索引显示
    frame_text = StringVar()
    frame_text.set(f"current frame: {frame_index}")
    frame_index_label = Label(root, textvariable=frame_text)
    frame_index_label.grid(row=1, column=0, columnspan=7)

    # 初始化选择状态显示
    child_status_text = StringVar()
    parent_status_text = StringVar()
    child_status_text.set("Child Label: None")
    parent_status_text.set("Parent Label: None")

    child_status_label = Label(root, textvariable=child_status_text)
    child_status_label.grid(row=3, column=0, columnspan=2)
    parent_status_label = Label(root, textvariable=parent_status_text)
    parent_status_label.grid(row=3, column=2, columnspan=2)

    # 创建标记按钮
    btn_child_true = Button(root, text="Child - True", command=mark_child_true)
    btn_child_true.grid(row=2, column=0)

    btn_child_false = Button(root, text="Child - False", command=mark_child_false)
    btn_child_false.grid(row=2, column=1)

    btn_parent_true = Button(root, text="Parent - True", command=mark_parent_true)
    btn_parent_true.grid(row=2, column=2)

    btn_parent_false = Button(root, text="Parent - False", command=mark_parent_false)
    btn_parent_false.grid(row=2, column=3)

    # 添加跳过和返回按钮
    Button(root, text="跳到下一个 8 帧", command=next_frame).grid(row=2, column=4)
    Button(root, text="返回上一个 8 帧", command=previous_frame).grid(row=2, column=5)

    # 加载第一帧
    update_frame()

    # 关闭程序时保存标记到CSV文件
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
