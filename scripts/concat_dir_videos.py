import os
import cv2
import numpy as np
from pathlib import Path

# Parameters
output_file = 'output_grid2.mp4'
grid_size = 12  # 8x8 grid

# Get sorted list of 64 video paths
dir_path = '/home/dell/temp2'
video_files = []
for dir in os.listdir(dir_path):
    if Path(dir_path, dir).is_dir():
        # load task_info.json to check success
        if 'failed' in dir:
            continue
        # task_info_path = Path(dir_path, dir, 'task_info.json')
        # with open(task_info_path, 'r') as f:
        #     import json
        #     task_info = json.load(f)
        #     if not task_info.get("success", False):
        #         continue
        # if "yellow" not in task_info["name"]:
        #     continue
        video_path = Path(dir_path, dir, 'main_image.mp4')
        if video_path.exists():
            video_files.append(str(video_path))
        if len(video_files) >= grid_size * grid_size:
            break

if len(video_files) < grid_size * grid_size:
    while len(video_files) < grid_size * grid_size:
        video_files.append(video_files[-1])
# Open video captures
caps = [cv2.VideoCapture(f) for f in video_files]

# Find minimum frame count and frame size
frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
min_frames = max(frame_counts)

widths = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in caps]
heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in caps]
min_width = 160
min_height = 120
max_width = max(widths)
max_height = max(heights)
fpss = [cap.get(cv2.CAP_PROP_FPS) for cap in caps]
min_fps = min(fpss)

# Prepare output video writer
out_width = 1920
out_height = 1080
# ... (rest of your code for loading videos, etc.)
import numpy as np

# ... (your code for loading videos, etc.)

# Animation parameters
stage1_frames = int(min_fps * 2)  # 5 seconds
stage2_frames = int(min_fps * 3)
stage3_frames = int(min_fps * 3)
ANIMATION_FRAMES = stage1_frames + stage2_frames + stage3_frames

medium_grid = 4  # 4x4 window

center_cell = grid_size // 2
cell_w = max_width
cell_h = max_height

out_width = 1920
out_height = 1080
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, min_fps, (out_width, out_height))
last_frames = [np.zeros((max_height, max_width, 3), dtype=np.uint8) for _ in range(len(caps))]
min_frames += 100

for frame_idx in range(min_frames):
    frames = []
    for i in range(len(caps)):
        cap = caps[i]
        ret, frame = cap.read()
        if not ret:
            frame = last_frames[i].copy()
            frame = cv2.addWeighted(frame, 0.5, np.zeros(frame.shape, frame.dtype), 0, 0)
        else:
            last_frames[i] = frame.copy()
        frame = cv2.resize(frame, (cell_w, cell_h))
        frames.append(frame)
    # Stack frames into grid
    grid_rows = []
    for i in range(grid_size):
        row = np.hstack(frames[i*grid_size:(i+1)*grid_size])
        grid_rows.append(row)
    grid_frame = np.vstack(grid_rows)
    grid_h, grid_w = grid_frame.shape[:2]

    # --- 3-STAGE ANIMATION ---
    if frame_idx < ANIMATION_FRAMES:
        # Stage split
        if frame_idx < stage1_frames:
            # --- Stage 1: Zoom from 1 cell to 4x4 window, centered ---
            t = frame_idx / (stage1_frames - 1)
            t = 1 - (1 - t) ** 2  # Ease-out

            # Start: 1 cell, End: 4x4 cells
            start_w = cell_w
            start_h = cell_h
            end_w = cell_w * medium_grid
            end_h = cell_h * medium_grid

            zoom_w = int(start_w + t * (end_w - start_w))
            zoom_h = int(start_h + t * (end_h - start_h))

            # Centered on center cell
            center_x = (center_cell + 0.5) * cell_w - 2500
            center_y = (center_cell + 0.5) * cell_h 

            x1 = int(center_x - zoom_w / 2)
            y1 = int(center_y - zoom_h / 2)
            x2 = x1 + zoom_w
            y2 = y1 + zoom_h

        elif frame_idx < stage1_frames + stage2_frames:
            # --- Stage 2: Slide 4x4 window horizontally, keep size ---
            t = (frame_idx - stage1_frames) / (stage2_frames - 1)
            t = 0.5 - 0.5 * np.cos(np.pi * t)  # Ease-in-out

            win_w = cell_w * medium_grid
            win_h = cell_h * medium_grid

            # Keep y centered on center cell
            center_y = (center_cell + 0.5) * cell_h 
            y1 = int(center_y - win_h / 2)
            y2 = y1 + win_h

            # Slide x from center to rightmost
            start_x = (center_cell + 0.5) * cell_w - win_w / 2 - 2500
            end_x = (center_cell + 0.5) * cell_w - win_w / 2
            x1 = int(start_x + t * (end_x - start_x))
            x2 = x1 + win_w

        else:
            # --- Stage 3: Zoom out from 4x4 at right edge to full grid, recenter to full grid ---
            t = (frame_idx - stage1_frames - stage2_frames) / (stage3_frames - 1)
            t = 1 - (1 - t) ** 2  # Ease-out

            # Start: 4x4 window at right edge; End: full grid centered
            start_w = cell_w * medium_grid
            start_h = cell_h * medium_grid
            end_w = grid_w
            end_h = grid_h

            # Start position: right edge, vertically centered
            start_x = (center_cell + 0.5) * cell_w - win_w / 2 
            start_y = (center_cell + 0.5) * cell_h - win_h / 2
            # End position: center of grid
            end_x = (grid_w - end_w) // 2
            end_y = (grid_h - end_h) // 2

            # Linear interpolate window size and position
            zoom_w = int(start_w + t * (end_w - start_w))
            zoom_h = int(start_h + t * (end_h - start_h))
            x1 = int(start_x + t * (end_x - start_x))
            y1 = int(start_y + t * (end_y - start_y))
            x2 = x1 + zoom_w
            y2 = y1 + zoom_h

        # Clamp
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(grid_w, x2)
        y2 = min(grid_h, y2)
        zoomed = grid_frame[y1:y2, x1:x2]
        grid_frame = cv2.resize(zoomed, (out_width, out_height))
    else:
        # Show full grid
        grid_frame = cv2.resize(grid_frame, (out_width, out_height))

    out.write(grid_frame)
    if frame_idx % 10 == 0:
        print(f'Processed frame {frame_idx+1}/{min_frames}')

# Release resources (same as your code)
for cap in caps:
    cap.release()
out.release()
print('Done! Output saved as', output_file)
