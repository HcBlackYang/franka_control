import os
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from cameras import camera_param
from cameras.camera_param import CameraParam
from robots import robot_param
from robots.robot_param import RobotParam


class StatesIterator:
    def __init__(self, states: list):
        self.states = states
        self.index = 0
        self.length = len(states)
    
    def reset(self):
        self.index = 0
    
    def get_next_state(self, timestamp: float):
        while self.index < self.length and self.states[self.index]['timestamp'] < timestamp:
            self.index += 1
        # cal diff
        time_diff = abs(self.states[self.index - 1]['timestamp'] - timestamp) if self.index > 0 else float('inf')
        if self.index == 0:
            return self.states[0]
        if self.index >= self.length:
            return self.states[-1]
        return self.states[self.index - 1]
    def sample_state(self, num: int, fps: float = 30, current_time: float = None):
        sampled_states = []
        interval = 1.0 / fps
        index = self.index
        current_time = self.states[self.index]['timestamp'] if current_time is None else current_time
        while len(sampled_states) < num:
            while index < self.length and self.states[index]['timestamp'] < current_time:
                index += 1
            if index == 0:
                state = self.states[0]
            elif index >= self.length:
                state = self.states[-1]
            else:
                
                state = self.states[index - 1]
            sampled_states.append(state)
            current_time += interval
        return sampled_states

def show_eval_dir(dir: str, show_traj: bool = True):
    main_image_path = Path(dir) / "main_image.mp4"
    main_image_cap = cv2.VideoCapture(str(main_image_path))
    if not main_image_cap.isOpened():
        print(f"Failed to open video file: {main_image_path}")
        return
    main_image_timestamps_path = Path(dir) / "main_image_timestamps.json"
    with open(main_image_timestamps_path, 'r') as f:
        main_image_timestamps = json.load(f)
    print(f"Total frames in main camera video: {int(main_image_cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    print(f"Total timestamps in main camera: {len(main_image_timestamps)}")
    assert int(main_image_cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(main_image_timestamps), "Frame count and timestamp count do not match!"
    frame_idx = 0
    
    eval_data_path = Path(dir) / "eval_data.json"
    with open(eval_data_path, 'r') as f:
        eval_data = json.load(f)
    eval_data_iter = StatesIterator(eval_data)
    eval_data_iter.reset()
    camera_param = CameraParam()
    camera_param.load_from_file(camera_name="main_image", file_dir="/home/dell/PARAM")
    robot_param = RobotParam()
    robot_param.load_from_file(file_dir="/home/dell/PARAM")

    robot_states_path = Path(dir) / "FrankaEmika_states.json"
    with open(robot_states_path, 'r') as f:
        robot_states = json.load(f)
    robot_states_iter = StatesIterator(robot_states)
    robot_states_iter.reset()
    action_count = 0

    while True:
        if frame_idx >= len(main_image_timestamps):
            break
        ret, frame = main_image_cap.read()
        frame_time = main_image_timestamps[frame_idx] 
        frame_idx += 1
        state = eval_data_iter.get_next_state(frame_time)
        robot_state = robot_states_iter.get_next_state(frame_time)
        print("Frame time:", frame_time)
        print(state['timestamp'])

        actions = np.array(state['actions'])
        actions_traj = actions[action_count//3:,:3].cumsum(axis=0) * 0.1 + np.array(robot_state['eef_pose'][:3])
        actions_traj_in_world = robot_param.transform_to_world(actions_traj)
        frame_to_show = frame.copy()

        if show_traj:
            state['trajectory'] = np.array(state['trajectory'])
            frame_to_show = camera_param.draw_trajectory_on_image(frame_to_show, state['trajectory'])
        frame_to_show = camera_param.draw_trajectory_on_image(frame_to_show, actions_traj_in_world)
        if not ret:
            break
        cv2.imshow('Main Camera', frame_to_show)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    main_image_cap.release()
    
    # use matplotlib to show the trajectory
    # save the frame and trajectory and actions_traj_in_world to json
    cv2.imwrite(Path(dir) / "final_frame.png", frame)
    traj_data = {
        "trajectory": state['trajectory'] if 'trajectory' in state else [],
        "actions_trajectory": actions_traj_in_world.tolist()
    }
    with open(Path(dir) / "show_data.json", 'w') as f:
        json.dump(traj_data, f, indent=4)
    

if __name__ == "__main__":
    # dir = "/home/dell/maple_control/data/evaluation/20250912_stack_rings_trajsteps50_fps15-5/20250912_210939"
    dir = "/home/dell/maple_control/data/evaluation/20250916_fruitsandtray_trajsteps50_fps30-10_200hand_407robot/20250916_200918"
    show_eval_dir(dir, show_traj=False)