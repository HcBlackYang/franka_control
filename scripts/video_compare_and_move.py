import os
from pathlib import Path
import random
import json
import shutil

dir1 = "/home/dell/maple_control/data/evaluation/20250922_fruitsandtray_trajsteps50_fps30-10_650hand_195bluerobot"
dir2 = r"/media/dell/T5 EVO/backup/evaluation/20250911_fps30-10"

dir1_destination = "/home/dell/eval_video_for_paper/fruits2/baseline"
dir2_destination = "/home/dell/eval_video_for_paper/fruits2/ours"

files_dir1 = os.listdir(dir1)
files_dir2 = os.listdir(dir2)

files_dir1.sort()
files_dir2.sort()


def save_video(eval_id):
    EVAL_DATA= "eval_info.json"
    with open(Path(dir1) / Path(files_dir1[eval_id]) / EVAL_DATA, 'r') as f:
        eval_data = json.load(f)
        success1 = eval_data.get("success", False)
    with open(Path(dir2) / Path(files_dir2[eval_id]) / EVAL_DATA, 'r') as f:
        eval_data = json.load(f)
        success2 = eval_data.get("success", False)
    if success2 != False:
        return
    # move the whole two folders to destination using shutil for better error handling

    src1 = Path(dir1) / files_dir1[eval_id]
    dst1 = Path(dir1_destination) / files_dir1[eval_id]
    src2 = Path(dir2) / files_dir2[eval_id]
    dst2 = Path(dir2_destination) / files_dir2[eval_id]

    try:
        shutil.copytree(src1, dst1, dirs_exist_ok=True)
        print(f"cp -r {src1} {dst1}")
    except Exception as e:
        print(f"Error copying {src1} to {dst1}: {e}")

    try:
        shutil.copytree(src2, dst2, dirs_exist_ok=True)
        print(f"cp -r {src2} {dst2}")
    except Exception as e:
        print(f"Error copying {src2} to {dst2}: {e}")

    

if __name__ == "__main__":
    eval_ids = list(range(len(files_dir1)))
    for i in range(len(eval_ids)):
        try:
            save_video(i)
        except Exception as e:
            print(f"Error processing eval id {i}: {e}")
        # if i >= 5:
        #     break