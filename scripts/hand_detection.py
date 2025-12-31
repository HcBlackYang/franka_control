import numpy as np
import cv2
import json
import os
from tqdm import tqdm
import websocket

def hand_detect(data_path="/home/dell/maple_control/data/stack_rings_hand/20250902_164607"):
    print("Start processing files in:", data_path)
    ws = websocket.create_connection("ws://10.21.40.5:8765", timeout=600)
    files = [f for f in os.listdir(data_path) if f.endswith('.mp4')]
    image_names = [
        "top_image",
        "main_image",
        "wrist_image"
    ]
    if not files:
        print("No .mp4 files found in the specified directory.")
        ws.close()
        return

    for file in files:
        detections = []
        cap = cv2.VideoCapture(os.path.join(data_path, file))
        camera_name = file.replace(".mp4", "")
        if camera_name not in image_names:
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing camera: {camera_name} with {total_frames} frames.")

        with tqdm(total=total_frames, desc=f"Processing {camera_name}", unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print(f"End of video file {file}.")
                    break
                _, img_encoded = cv2.imencode('.jpg', frame)
                ws.send_binary(img_encoded.tobytes())
                result = ws.recv()
                detection_result = json.loads(result)
                if "error" in detection_result:
                    print(f"Error: {detection_result['error']}")
                    pbar.update(1)
                    continue
                detections.append(detection_result)
                pbar.update(1)
        output_file = os.path.join(data_path, f"{os.path.splitext(file)[0]}_detections.json")
        with open(output_file, 'w') as f:
            json.dump(detections, f, indent=4)
        print(f"Detections saved to {output_file}")
        cap.release()
    ws.close()

if __name__ == "__main__":
    hand_detect(data_path="/home/dell/maple_control/data/stack_rings_hand/20250902_222423")