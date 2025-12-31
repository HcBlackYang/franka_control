import cv2
from pathlib import Path
import os
from cameras.camera_param import CameraParam
from robots.robot_param import RobotParam
import numpy as np
import logging

ROBOT_STATS = "FrankaEmika_states.json"
IMAGE_TIMESTAMPS = [
    "main_image_timestamps.json",
    "wrist_image_timestamps.json"
]
IMAGE_VIDEO = [
    "main_image.mp4",
    "wrist_image.mp4"
]
CONTROLLER_STATS = "SpaceMouseController_states.json"
TASK_INFO = "task_info.json"
ROBOT_PARAM = "robot_param.json"
CAMERA_PARAM = [
    "default_image_camera_param.json",
]

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
        

class DataReplayer:
    def __init__(self, data_path: str, fps: float = 30):
        self.data_path = Path(data_path)
        self.fps = fps
        self.robot_states = []
        self.controller_states = []
        self.task_info = {}
        self.robot_param = None
        self.camera_params = {
            "main_image": None,
        }
        self.cams = {
            "main_image": None,
            "wrist_image": None
        }
        self.image_timestamps = {
            "main_image": [],
            "wrist_image": []
        }
        self.load_data()
        self.robot_states_iterator = StatesIterator(self.robot_states)
        self.controller_states_iterator = StatesIterator(self.controller_states)
        self.start_timestamp = self.check_max_timestamp()

    def load_data(self):
        import json
        # Load robot states
        robot_stats_path = self.data_path / ROBOT_STATS
        if robot_stats_path.exists():
            with open(robot_stats_path, 'r') as f:
                self.robot_states = json.load(f)
        else:
            raise FileNotFoundError(f"Robot stats file not found at {robot_stats_path}")
        
        # Load controller states
        controller_stats_path = self.data_path / CONTROLLER_STATS
        if controller_stats_path.exists():
            with open(controller_stats_path, 'r') as f:
                self.controller_states = json.load(f)
        else:
            raise FileNotFoundError(f"Controller stats file not found at {controller_stats_path}")
        # Load task info
        task_info_path = self.data_path / TASK_INFO
        if task_info_path.exists():
            with open(task_info_path, 'r') as f:
                self.task_info = json.load(f)
        else:
            raise FileNotFoundError(f"Task info file not found at {task_info_path}")
        
        # Load robot param
        robot_param_path = self.data_path / ROBOT_PARAM
        if robot_param_path.exists():
            self.robot_param = RobotParam()
            self.robot_param.load_from_file(str(self.data_path))
        else:
            raise FileNotFoundError(f"Robot param file not found at {robot_param_path}")
        
        # Load camera params
        self.camera_params["main_image"] = CameraParam()
        self.camera_params["main_image"].load_from_file(str(self.data_path))

        # Load image timestamps
        for ts_file in IMAGE_TIMESTAMPS:
            ts_path = self.data_path / ts_file
            if ts_path.exists():
                with open(ts_path, 'r') as f:
                    self.image_timestamps[str(ts_file).replace("_timestamps.json", "")] = json.load(f)
            else:
                raise FileNotFoundError(f"Image timestamps file not found at {ts_path}")
        
        for cam_file in IMAGE_VIDEO:
            cam_path = self.data_path / cam_file
            if cam_path.exists():
                cap = cv2.VideoCapture(str(cam_path))
                if not cap.isOpened():
                    raise IOError(f"Cannot open video file {cam_path}")
                self.cams[cam_file.replace(".mp4", "")] = cap
            else:
                raise FileNotFoundError(f"Camera video file not found at {cam_path}")
        # check if timestamps length match video frames
        for cam_name in self.cams:
            if self.cams[cam_name] is not None:
                frame_count = int(self.cams[cam_name].get(cv2.CAP_PROP_FRAME_COUNT))
                ts_count = len(self.image_timestamps[cam_name])
                if frame_count != ts_count:
                    raise ValueError(f"Frame count {frame_count} does not match timestamp count {ts_count} for camera {cam_name}")
                
    def check_max_timestamp(self):
        max_timestamp = 0
        
        if self.robot_states[0]['timestamp'] > max_timestamp:
            max_timestamp = self.robot_states[0]['timestamp']
        if self.controller_states[0]['timestamp'] > max_timestamp:
            max_timestamp = self.controller_states[0]['timestamp']
        for cam in self.image_timestamps:
            if len(self.image_timestamps[cam]) > 0 and self.image_timestamps[cam][0] > max_timestamp:
                max_timestamp = self.image_timestamps[cam][0]
        return max_timestamp

    def replay(self):
        frame_index = 0
        while True:
            
            if frame_index >= len(self.image_timestamps["main_image"]) or frame_index >= len(self.image_timestamps["wrist_image"]):
                break

            frame_time_diff = self.image_timestamps["main_image"][frame_index] - self.image_timestamps["wrist_image"][frame_index]
            frame_average_time = self.image_timestamps["main_image"][frame_index] 
            robot_states = self.robot_states_iterator.get_next_state(frame_average_time)
            controller_states = self.controller_states_iterator.get_next_state(frame_average_time)
            traj_show_fps = 10
            controller_states_sampled = self.controller_states_iterator.sample_state(50, fps=traj_show_fps)
            trajectory = []
            for state in controller_states_sampled:
                last_point = None
                if len(trajectory) > 0:
                    last_point = trajectory[-1]
                else:
                    last_point = robot_states['eef_pose'][:3]
                trajectory.append(
                    np.array(state['action'][:3] ) * (1/traj_show_fps) + last_point
                    )

            trajectory = np.array(trajectory)
            if abs(frame_time_diff) > 0.060:
                logging.error(f"Frame time difference too large: {frame_time_diff}")
                break
            frame_index += 1
            for cam_name, cam in self.cams.items():
                if cam is None:
                    raise ValueError(f"Camera {cam_name} is not loaded properly.")
                ret, frame = cam.read()
                if not ret:
                    print("End of video stream")
                    return
                eef_pose = np.array(robot_states['eef_pose'])
                if cam_name == "main_image":
                    frame_to_show = frame.copy()
                    trajectory = self.robot_param.transform_to_world(trajectory)
                    if self.camera_params["main_image"] is not None:
                        frame_to_show = self.camera_params["main_image"].draw_trajectory_on_image(frame_to_show, trajectory)
                    cv2.imshow(f'{cam_name}', frame_to_show)
                else:
                    cv2.imshow(f'{cam_name}', frame)
                    if cv2.waitKey(int(1000 / self.fps)) & 0xFF == ord('q'):
                        break

            
                    
if __name__ == "__main__":
    data_replay = DataReplayer("/home/dell/maple_control/data/20250829_fruits_and_tray/20250829_170521")
    print(f"Loaded {len(data_replay.robot_states)} robot states")
    print(f"Loaded {len(data_replay.controller_states)} controller states")
    print(f"Task info: {data_replay.task_info}")
    print(f"Robot param: {data_replay.robot_param}")
    print(f"Loaded {len(data_replay.camera_params)} camera params")
    print(f"Start timestamp: {data_replay.start_timestamp}")
    data_replay.replay()