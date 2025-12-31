import json
from pathlib import Path
import os

def check_gripper_state(data_dir: str):
    ROBOT_STATE_FILE = "SpaceMouseController_states.json"
    with open(Path(data_dir) / ROBOT_STATE_FILE, 'r') as f:
        robot_states = json.load(f)
    for state in robot_states:
        print(state['gripper']['target_position'])

if __name__ == "__main__":
    check_gripper_state("/home/dell/maple_control/data/20250829_fruits_and_tray/20250829_170437")