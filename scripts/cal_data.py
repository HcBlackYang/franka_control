import os
from pathlib import Path
import json



def cal_states(dir: str = None):
    CALIB_FILE_NAME= "task_info.json"
    task_states = {}

    for file in os.listdir(dir):
        calib_data_path = Path(dir) / Path(file) / CALIB_FILE_NAME
        if 'failed' in file:
            continue
        if calib_data_path.exists():
            with open(calib_data_path, 'r') as f:
                calib_data = json.load(f)
                task_name = calib_data.get("name", "unknown_task")
                if not task_name in task_states:
                    task_states[task_name] = 0
                task_states[task_name] += 1
        else:
            print(f"No calibration data found for {file}")    
    print("\nTask states summary:")
    for task, count in task_states.items():
        print(f"Task: {task}, Count: {count}")

if __name__ == "__main__":
    cal_states("/home/dell/maple_control/data/20250829_fruits_and_tray")