import cv2
from pathlib import Path
import os
from cameras.camera_param import CameraParam
from robots.robot_param import RobotParam
import numpy as np
import logging


IMAGE_NAME = [
    "main_image",
    "wrist_image",
    "top_image",
    "human_image"
]
TASK_INFO = "task_info.json"
HAND_POSE = "hand_mano_pose_estimation_results.json"

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
        
POINTS_MAP = {
        0:0, 5:1, 6:2, 7:3, 9:4, 10:5, 11:6, 17:7, 18:8,
        19:9,
        13:10,
        14:11,
        15:12,
        1:13,
        2:14,
        3:15,
        4:16,
        8:17,
        12:18,
        16:19,
        20:20
    }
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
        self.cam_shape = {

        }
        self.image_timestamps = {
            "main_image": [],
            "wrist_image": []
        }
        self.image_detections_origin = {
            "main_image": [],
            "wrist_image": []
        }
        self.image_detections = {
        }
        self.load_data()

    def load_data(self):
        import json
        # Load robot states
      
        # Load task info
        task_info_path = self.data_path / TASK_INFO
        if task_info_path.exists():
            with open(task_info_path, 'r') as f:
                self.task_info = json.load(f)
        else:
            raise FileNotFoundError(f"Task info file not found at {task_info_path}")
        # Load camera params
        for image_name in IMAGE_NAME:
            if Path.exists(self.data_path / f"{image_name}_camera_param.json"):
                self.camera_params[image_name] = CameraParam()
                self.camera_params[image_name].load_from_file(str(self.data_path), camera_name=image_name)
            
            cam_path = self.data_path / f"{image_name}.mp4"
            if cam_path.exists():
                cap = cv2.VideoCapture(str(cam_path))
                if not cap.isOpened():
                    raise IOError(f"Cannot open video file {cam_path}")
                self.cams[image_name] = cap
                self.cam_shape[image_name] = (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )
            
            timestamps_path = self.data_path / f"{image_name}_timestamps.json"
            if timestamps_path.exists():
                with open(timestamps_path, 'r') as f:
                    self.image_timestamps[image_name] = json.load(f)
            
            detections_path = self.data_path / f"{image_name}_detections.json"
            if detections_path.exists():
                with open(detections_path, 'r') as f:
                    detections = json.load(f)
                self.image_detections_origin[image_name] = detections
        self.load_detections()
        # Load hand pose estimation results
        hand_pose_path = self.data_path / HAND_POSE
        if hand_pose_path.exists():
            with open(hand_pose_path, 'r') as f:
                self.hand_pose = json.load(f)
        else:
            self.hand_pose = None
            logging.warning(f"Hand pose estimation file not found at {hand_pose_path}, proceeding without it.")
    def draw_hand(self, frame, hand_detection):
        if hand_detection is not None:
            if len(hand_detection) > 0:
                self.draw_points(frame, hand_detection, color=(0, 255, 0), radius=5)
                if len(hand_detection) > 16:
                    self.draw_points(frame, 
                                    [((np.array(hand_detection[16])+np.array(hand_detection[17]))/2).tolist()], 
                                    color=(0, 0, 255), radius=8)

    def draw_points(self, image, points, color=(0, 255, 0), radius=5):
        for point in points:
            if point is None or len(point) != 2:
                continue
            x = int(point[0])
            y = int(point[1])
            cv2.circle(image, (x, y), radius, color, -1)
        return image
    def load_detections(self):
        for cam_name, detections in self.image_detections_origin.items():
            assert len(detections) == len(self.image_timestamps[cam_name]), f"Detections and timestamps length mismatch for {cam_name}, {len(detections)} vs {len(self.image_timestamps[cam_name])}"
            self.image_detections[cam_name] = []
            frame_index = 0
            for detection in detections:
                target_index = -1
                max_score = -1
                self.image_detections[cam_name].append([])
                for index in range(len(detection['handedness'])):
                    if detection['handedness'][index]['score'] > max_score:
                        target_index = index
                        max_score = detection['handedness'][index]['score']
                if target_index == -1:
                    self.image_detections[cam_name][frame_index].append(None)
                else:
                    self.image_detections[cam_name][frame_index] = np.zeros((21, 2)).tolist()
                    for i in range(len(detection['hand_landmarks'][target_index])):
                        self.image_detections[cam_name][frame_index][POINTS_MAP[i]] = [
                            detection['hand_landmarks'][target_index][i]['x'] * self.cam_shape[cam_name][0],
                            detection['hand_landmarks'][target_index][i]['y'] * self.cam_shape[cam_name][1]
                        ]              
                frame_index += 1
  
    def replay(self):
        frame_index = 0
        while True:
            if frame_index >= len(self.image_timestamps["main_image"]) or frame_index >= len(self.image_timestamps["wrist_image"]):
                break
            frame_time_diff = self.image_timestamps["main_image"][frame_index] - self.image_timestamps["wrist_image"][frame_index]
            frame_average_time = self.image_timestamps["main_image"][frame_index] 
            if abs(frame_time_diff) > 0.160:
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
                # self.draw_hand(frame, self.image_detections.get(cam_name, [None]*len(self.image_timestamps[cam_name]))[frame_index-1])
                if self.hand_pose is not None:
                    hand_pose = self.hand_pose['joints_3d_batch'][frame_index-1]
                    if hand_pose is not None and cam_name in self.camera_params:
                        frame = self.camera_params[cam_name].draw_trajectory_on_image(frame, hand_pose)
                        frame = self.camera_params[cam_name].draw_trajectory_on_image(frame, 
                                                                                      [((np.array(hand_pose[16])+np.array(hand_pose[17]))/2).tolist()],)
                cv2.imshow(f"Camera: {cam_name}", frame)
            cv2.waitKey(0)
                    
if __name__ == "__main__":
    data_replay = DataReplayer("/home/dell/maple_control/data/20250914_stack_rings_hand/20250913_165758")

    data_replay.replay()