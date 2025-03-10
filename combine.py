import re
import os
import json

BASE_DIR = os.getenv('BASE_DIR', '/app/Desktop/Dataset/3')
VIDEO_NAME = os.path.basename(BASE_DIR)

# 假设从文本文件加载对话
dialogue_file_path = os.path.join(BASE_DIR, f'{VIDEO_NAME}.srt')
gaze_events_json_path = os.path.join(BASE_DIR, f'{VIDEO_NAME}_gaze_events.json')
output_json_path = os.path.join(BASE_DIR, f'{VIDEO_NAME}_result.json')  # 输出 JSON 文件路径

with open(gaze_events_json_path, 'r') as f:
    gaze_events = json.load(f)

# 解析对话文件，提取时间戳和文本
def parse_dialogue_file(file_path):
    dialogue_lines = []
    block = {}  # 保存一个对话块的信息

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            # 如果是编号行，初始化一个新的对话块
            if re.match(r'^\d+$', line):
                if block:  # 如果已经有一个完整的块，保存它
                    dialogue_lines.append(block)
                block = {'id': line}

            # 匹配时间戳的行
            elif re.match(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', line):
                block['time'] = line  # 保存时间戳

            # 剩下的是对话文本
            elif line:
                if 'text' in block:
                    block['text'] += " " + line  # 将多行文本合并
                else:
                    block['text'] = line

        # 将最后一个块添加到列表中
        if block:
            dialogue_lines.append(block)

    return dialogue_lines

# 解析时间戳函数，将格式 "00:00:10,759" 转为秒
def parse_timestamp(timestamp):
    hours, minutes, seconds, milliseconds = re.match(r"(\d+):(\d+):(\d+),(\d{3})", timestamp).groups()
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000.0
    return total_seconds

# 匹配 gaze 事件和对话时间，按时间线输出
def match_gaze_events_with_dialogue(gaze_events, dialogue_lines):
    output = []  # 保存最终的输出结果
    gaze_event_idx = 0  # gaze_events的指针

    for dialogue in dialogue_lines:
        start_time, end_time = re.match(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", dialogue['time']).groups()
        start_seconds = parse_timestamp(start_time)
        end_seconds = parse_timestamp(end_time)

        dialogue_entry = {
            "start_time": start_time,
            "end_time": end_time,
            "text": dialogue['text'],
            "gaze_events": []
        }

        # 依次检查 gaze_events 是否在当前对话的时间范围内
        while gaze_event_idx < len(gaze_events) and gaze_events[gaze_event_idx] <= end_seconds:
            event_time = gaze_events[gaze_event_idx]
            if start_seconds <= event_time <= end_seconds:
                dialogue_entry['gaze_events'].append(event_time)
            gaze_event_idx += 1

        # 无论是否有匹配的 gaze_events，都输出当前的对话
        output.append(dialogue_entry)

    return output

# 读取对话文件并解析
dialogue_lines = parse_dialogue_file(dialogue_file_path)

# 运行匹配函数，按时间线输出
output = match_gaze_events_with_dialogue(gaze_events, dialogue_lines)

# 将最终结果保存到 JSON 文件中
with open(output_json_path, 'w') as f:
    json.dump(output, f, indent=4)

print(f"结果已成功保存到 {output_json_path}")
