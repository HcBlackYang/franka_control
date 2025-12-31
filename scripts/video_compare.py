import os
from pathlib import Path
import json
import cv2
import numpy as np
import subprocess # Import the subprocess module

# dir1 = "/home/dell/maple_control/data/evaluation/20250907_pickup_bottle_200epi_baseline_100k"
# dir2 = "/home/dell/maple_control/data/evaluation/20250907_pickup_bottle_200epi_cotrain_100k"
dir1 = "/home/dell/eval_video_for_paper/rings/baseline"
dir2 = "/home/dell/eval_video_for_paper/rings/ours"

files_dir1 = os.listdir(dir1)
files_dir2 = os.listdir(dir2)

files_dir1.sort()
files_dir2.sort()

def compress_video(input_path, output_path):
    """
    Compresses a video using FFmpeg to 480p, 15fps, and a lower bitrate.
    """
    print(f"Compressing {input_path} to {output_path}...")
    try:
        # FFmpeg command:
        # -i: input file
        # -vf "scale=-2:480": resize video to 480p height, maintaining aspect ratio
        # -r 15: set frame rate to 15 fps
        # -c:v libx264: use the x264 video codec
        # -crf 28: set Constant Rate Factor for compression (higher value = smaller file)
        # -preset veryfast: encoding speed preset
        # -c:a aac: use the aac audio codec
        # -b:a 128k: set audio bitrate
        # -y: overwrite output file if it exists
        command = [
            'ffmpeg',
            '-i', str(input_path),
            '-vf', 'scale=-2:480',
            '-r', '30',
            '-c:v', 'libx264',
            '-crf', '28',
            '-preset', 'veryfast',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-y',
            str(output_path)
        ]
        # Execute the command
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully compressed video to {output_path}")
        # Clean up the original, larger file
        os.remove(input_path)
        print(f"Removed original file: {input_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg compression for {input_path}:")
        print(f"FFmpeg stdout: {e.stdout}")
        print(f"FFmpeg stderr: {e.stderr}")
        raise


def save_video(eval_id):
    EVAL_DATA = "eval_info.json"
    with open(Path(dir1) / Path(files_dir1[eval_id]) / EVAL_DATA, 'r') as f:
        eval_data = json.load(f)
        success1 = eval_data.get("success", False)
    with open(Path(dir2) / Path(files_dir2[eval_id]) / EVAL_DATA, 'r') as f:
        eval_data = json.load(f)
        success2 = eval_data.get("success", False)
    print(f"Processing eval id {eval_id}, {files_dir1[eval_id]} vs {files_dir2[eval_id]} - Success: {success1} vs {success2}")

    if success1:
        return

    MAIN_VIDEO = "main_image.mp4"
    print(f"Start saving comparison video for eval id {eval_id}, {files_dir1[eval_id]} vs {files_dir2[eval_id]}")
    video1_path = Path(dir1) / Path(files_dir1[eval_id]) / MAIN_VIDEO
    video2_path = Path(dir2) / Path(files_dir2[eval_id]) / MAIN_VIDEO

    cap1 = cv2.VideoCapture(str(video1_path))
    cap2 = cv2.VideoCapture(str(video2_path))
    
    # Use the original FPS for writing, compression will handle the change to 15fps
    fps = min(cap1.get(cv2.CAP_PROP_FPS), cap2.get(cv2.CAP_PROP_FPS))
    
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_width = width1 + width2
    out_height = max(height1, height2)
    
    # --- Define output filenames ---
    uncompressed_output_filename = f'temp_comparison_video_{eval_id}.mp4'
    compressed_output_filename = f'comparison_video_{eval_id}_480p.mp4'
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(uncompressed_output_filename, fourcc, fps, (out_width, out_height))

    # Define text properties for a larger, bolder 'x3'
    text = "x3"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    font_color = (255, 255, 255)
    thickness = 12

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    while True:
        # Skip 2 out of every 3 frames to achieve 3x speed
        for _ in range(2):
            cap1.grab()
            cap2.grab()
        
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 and not ret2:
            break
        if not ret1:
            frame1 = np.zeros((height1, width1, 3), dtype=np.uint8)
        if not ret2:
            frame2 = np.zeros((height2, width2, 3), dtype=np.uint8)

        if height1 != out_height:
            frame1 = cv2.resize(frame1, (width1, out_height))
        if height2 != out_height:
            frame2 = cv2.resize(frame2, (width2, out_height))

        # Add original text (Success/Fail) to top-left
        color1 = (0, 255, 0) if success1 else (0, 0, 255)
        cv2.putText(frame1, f"Baseline - {'Success' if success1 else 'Fail'}", (20, 60), font, 1.5, color1, 4, cv2.LINE_AA)
        color2 = (0, 255, 0) if success2 else (0, 0, 255)
        cv2.putText(frame2, f"Ours - {'Success' if success2 else 'Fail'}", (20, 60), font, 1.5, color2, 4, cv2.LINE_AA)
        
        # Calculate position for the BOTTOM-RIGHT corner
        margin = 20
        pos1 = (width1 - text_width - margin, height1 - margin)
        cv2.putText(frame1, text, pos1, font, font_scale, font_color, thickness, cv2.LINE_AA)

        pos2 = (width2 - text_width - margin, height2 - margin)
        cv2.putText(frame2, text, pos2, font, font_scale, font_color, thickness, cv2.LINE_AA)

        combined_frame = np.hstack((frame1, frame2))
        out.write(combined_frame)

    cap1.release()
    cap2.release()
    out.release()
    print(f"Saved 3x speed comparison video to {uncompressed_output_filename}")

    # --- NEW: Add compression step ---
    compress_video(uncompressed_output_filename, compressed_output_filename)


if __name__ == "__main__":
    eval_ids = list(range(len(files_dir1)))
    for i in eval_ids:
        try:
            save_video(i)
        except Exception as e:
            print(f"Error processing eval id {i}: {e}")