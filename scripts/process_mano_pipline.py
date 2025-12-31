import os
import json
from pathlib import Path
from scripts.hand_pose_estimation import HandPoseEstimator

if __name__ == "__main__":
    dir = "/home/dell/maple_control/data/20250910_stack_cups_hand"
    error_files = []
    for file in os.listdir(dir):
        if file == 'failed_tasks':
            continue
        HAND_POSE_ESTIMATION_RESULTS = "hand_mano_pose_estimation_results.json"
        if os.path.exists(Path(dir) / Path(file) / HAND_POSE_ESTIMATION_RESULTS):
            print(f"{file} already processed, skip")
            continue
        print(f"Processing {file} ...")
        try:
            data_replay = HandPoseEstimator(str(Path(dir) / Path(file)))
            data_replay.replay()
        except:
            print(f"Error processing {file}, skip")
            error_files.append(file)
            continue
        print(f"Finished processing {file}.\n")
    print("Error files:")
    print(error_files)