import os
import cv2
import pandas as pd
from tkinter import Tk, Button, Label

# Load annotations or initialize if file does not exist
def load_annotations():
    global annotations, csv_path
    try:
        annotations = pd.read_csv(csv_path, index_col="frame")
    except FileNotFoundError:
        annotations = pd.DataFrame(columns=["frame", "child_label", "parent_label"]).set_index("frame")

# Update the current frame and display it with OpenCV
def update_frame():
    global frame_index, child_label, parent_label

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        return False  # Signal that video has ended

    # Load labels from annotations if available
    if frame_index in annotations.index:
        child_label = annotations.loc[frame_index, "child_label"] == "True"
        parent_label = annotations.loc[frame_index, "parent_label"] == "True"
    else:
        child_label, parent_label = None, None

    # Display frame in an OpenCV window
    cv2.imshow("Video Frame", frame)
    print(f"Displaying frame: {frame_index} | Child Label: {child_label} | Parent Label: {parent_label}")
    
    return True  # Signal that frame was updated successfully

# Save annotations to CSV
def save_annotation():
    if child_label is not None and parent_label is not None:
        annotations.loc[frame_index] = [child_label, parent_label]
        annotations.to_csv(csv_path, index_label="frame")

# Label functions for Tkinter buttons
def mark_child_true():
    global child_label
    child_label = True
    save_annotation()
    update_status_text()

def mark_child_false():
    global child_label
    child_label = False
    save_annotation()
    update_status_text()

def mark_parent_true():
    global parent_label
    parent_label = True
    save_annotation()
    update_status_text()

def mark_parent_false():
    global parent_label
    parent_label = False
    save_annotation()
    update_status_text()

def next_frame():
    global frame_index
    frame_index += 8
    update_frame()

def previous_frame():
    global frame_index
    frame_index = max(0, frame_index - 8)
    update_frame()

def update_status_text():
    child_status_label.config(text=f"Child Label: {child_label}")
    parent_status_label.config(text=f"Parent Label: {parent_label}")
    frame_index_label.config(text=f"Current Frame: {frame_index}")

def on_closing():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

def main():
    global cap, frame_index, child_label, parent_label, annotations, csv_path
    global child_status_label, parent_status_label, frame_index_label, root

    video_path = "/home/weiyan/Desktop/Dataset_mp4/Piaget - Object permanence failure (Sensorimotor Stage)/Piaget - Object permanence failure (Sensorimotor Stage).mp4"
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = f"/home/weiyan/Desktop/Dataset_mp4/{video_name}/{video_name}_annotations.csv"
    video_path = "/home/weiyan/Desktop/Dataset_mp4/Piaget - Object permanence failure (Sensorimotor Stage)/Piaget - Object permanence failure (Sensorimotor Stage)-gaze.mp4"
    
    load_annotations()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    frame_index = 0
    child_label, parent_label = None, None

    # Tkinter window setup
    root = Tk()
    root.title("Label Tool")

    # Create Tkinter labels and buttons
    frame_index_label = Label(root, text=f"Current Frame: {frame_index}")
    frame_index_label.grid(row=0, column=0, columnspan=4)

    child_status_label = Label(root, text=f"Child Label: {child_label}")
    child_status_label.grid(row=1, column=0, columnspan=2)
    parent_status_label = Label(root, text=f"Parent Label: {parent_label}")
    parent_status_label.grid(row=1, column=2, columnspan=2)

    Button(root, text="Next Frame (Skip 8)", command=next_frame).grid(row=2, column=0)
    Button(root, text="Previous Frame (Back 8)", command=previous_frame).grid(row=2, column=1)
    Button(root, text="Child True", command=mark_child_true).grid(row=3, column=0)
    Button(root, text="Child False", command=mark_child_false).grid(row=3, column=1)
    Button(root, text="Parent True", command=mark_parent_true).grid(row=3, column=2)
    Button(root, text="Parent False", command=mark_parent_false).grid(row=3, column=3)

    # Update frame in OpenCV window
    update_frame()

    # Set Tkinter to handle window close
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
