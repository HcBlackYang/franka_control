import os
import json
from pathlib import Path
from scripts.hand_detection import hand_detect

if __name__ == "__main__":
    dir = "/home/dell/maple_control/data/20250910_stack_cups_hand"
    for file in os.listdir(dir):
        CAMERA_NAMES = [
            "top_image",
            "main_image",
            "wrist_image"
        ]
        processed = True
        for cam_name in CAMERA_NAMES:
            if os.path.exists(str(Path(dir) / Path(file) / Path(f"{cam_name}_detections.json"))) is False:
                processed = False
        if processed:
            print(f"Skipping {file}, already processed.")
            continue
        print(f"Processing {file} ...")
        data_replay = hand_detect(str(Path(dir) / Path(file)))
        print(f"Finished processing {file}.\n")