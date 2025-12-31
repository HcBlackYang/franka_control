import cv2
from pathlib import Path
import os
from cameras.camera_param import CameraParam
from robots.robot_param import RobotParam
import numpy as np
import logging
import websockets
import json
import websocket

IMAGE_NAME = [
    "main_image",
    "wrist_image",
    "top_image",
    "human_image"
]
TASK_INFO = "task_info.json"


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
        # 手腕
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
class HandPoseEstimator:

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
        self.frames = {

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
        self.fram_len = 1000000
        self.load_data()
        uri = "ws://10.21.40.5:8766"
        self.ws = websocket.create_connection(
                url=uri, timeout=600
            )
        self.hand_pose = {
            "joints_3d_batch": [],
            "betas": [],
            "global_orient": [],
            "hand_pose": [],
            "transl": [],
            "loss": []
        }
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
            else:
                logging.warning(f"Camera param file for {image_name} not found.")
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
            if len(self.image_timestamps[image_name]) < self.fram_len:
                self.fram_len = len(self.image_timestamps[image_name])
        self.load_detections()
    
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
        # Distort points
        for cam_name, cam_param in self.camera_params.items():
            if cam_param is not None and cam_name in self.image_detections:
                for frame_idx in range(len(self.image_detections[cam_name])):
                    if self.image_detections[cam_name][frame_idx] is None or len(self.image_detections[cam_name][frame_idx]) == 0:
                        continue
                    if len(self.image_detections[cam_name][frame_idx]) != 21:
                        continue
                    points = np.array(self.image_detections[cam_name][frame_idx], dtype=np.float32)
                    points = points.reshape(-1, 1, 2)
                    undistorted_points = cv2.undistortPoints(points, cam_param.intrinsic_matrix, cam_param.distortion_coeffs, P=cam_param.intrinsic_matrix)
                    undistorted_points = undistorted_points.reshape(-1, 2)
                    self.image_detections[cam_name][frame_idx] = undistorted_points.tolist()
            
    
    def process_batch(self, batch):
        batch_size = 350
        # Split batch into smaller batches
        small_batches = []
        for i in range(0, len(batch), batch_size):
            small_batches.append(batch[i:i+batch_size])
        results = []

        for small_batch in small_batches:
            request_data = {"batch": small_batch}
            print("Start processing a batch of size:", len(small_batch))
            self.ws.send(json.dumps(request_data))
            response = self.ws.recv()
            result = json.loads(response)
            results.append(result)
        
        # combine results
        # "joints_3d_batch": [j.detach().cpu().numpy().tolist() for j in output.joints.detach()],
        # "betas": output.betas.detach().cpu().numpy().tolist(),
        # "global_orient": output.global_orient.detach().cpu().numpy().tolist(),
        # "hand_pose": output.hand_pose.detach().cpu().numpy().tolist(),
        # "transl": transl.detach().cpu().numpy().tolist(),
        # "loss": loss,
        combined_results = {
            "joints_3d_batch": [],
            "betas": [],
            "global_orient": [],
            "hand_pose": [],
            "transl": [],
            "loss": []
        }
        if 'error' in results[0]:
            print(f"Error from server: {results[0]['error']}")
            return results
        for res in results:
            combined_results["joints_3d_batch"].extend(res["joints_3d_batch"])
            combined_results["betas"].extend(res["betas"])
            combined_results["global_orient"].extend(res["global_orient"])
            combined_results["hand_pose"].extend(res["hand_pose"])
            combined_results["transl"].extend(res["transl"])
            combined_results["loss"].append(res["loss"])
        return combined_results

    def replay(self):
        frame_index = 0
        while True:
            if frame_index >= len(self.image_timestamps["main_image"]) or frame_index >= len(self.image_timestamps["wrist_image"]):
                break

            frame_time_diff = self.image_timestamps["main_image"][frame_index] - self.image_timestamps["wrist_image"][frame_index]
            frame_average_time = self.image_timestamps["main_image"][frame_index] 
        

            if abs(frame_time_diff) > 0.060:
                logging.error(f"Frame time difference too large: {frame_time_diff}")
                break
            for cam_name, cam in self.cams.items():
                if self.frames.get(cam_name) is None:
                    self.frames[cam_name] = []
                if cam is None:
                    raise ValueError(f"Camera {cam_name} is not loaded properly.")
                ret, frame = cam.read()
                if not ret:
                    print("End of video stream")
                    break
                self.frames[cam_name].append(frame.copy())

            frame_index += 1

        frame_index = 0
        batch = []
        missing_frame = []
        while True:
            frame_data = {
                "keypoints_2d": [],
                "K_list": [],
                "RT_list": [],
                "num_cams": 0
            }
            if frame_index >= self.fram_len:
                break
            for cam_name, cam in self.cams.items():

                if self.image_detections.get(cam_name) is None or self.image_detections[cam_name][frame_index] is None or len(self.image_detections[cam_name][frame_index]) != 21:
                    continue

                frame_data["keypoints_2d"].append(self.image_detections[cam_name][frame_index])
                frame_data["K_list"].append(self.camera_params[cam_name].intrinsic_matrix.tolist())
                #                         RT = np.hstack((R, np.array(params['tvec']).reshape(3, 1)))
                R, _ = cv2.Rodrigues(np.array(self.camera_params[cam_name].extrinsic_rvec))
                RT = np.hstack((R, np.array(self.camera_params[cam_name].extrinsic_tvec).reshape(3, 1)))
                frame_data["RT_list"].append(RT.tolist())
            if len(frame_data["keypoints_2d"]) < 2:
                missing_frame.append(frame_index)
            else:
                frame_data["num_cams"] = len(frame_data["keypoints_2d"])
                batch.append(frame_data)
            frame_index += 1
        result = self.process_batch(batch)
        if 'joints_3d_batch' not in result and 'error' in result[0]:
            print(f"Error from server: {result[0]['error']}")
            return
        cnt = 0
        for i in range(self.fram_len):
            if i in missing_frame:
                self.hand_pose['joints_3d_batch'].append(None)
                self.hand_pose['betas'].append(None)
                self.hand_pose['global_orient'].append(None)
                self.hand_pose['hand_pose'].append(None)
                self.hand_pose['transl'].append(None)
            else:
                self.hand_pose['joints_3d_batch'].append(result['joints_3d_batch'][cnt])
                self.hand_pose['betas'].append(result['betas'][cnt])
                self.hand_pose['global_orient'].append(result['global_orient'][cnt])
                self.hand_pose['hand_pose'].append(result['hand_pose'][cnt])
                self.hand_pose['transl'].append(result['transl'][cnt])
                cnt += 1
        self.hand_pose['loss'] = result['loss']
        pose_path = self.data_path / "hand_mano_pose_estimation_results.json"
        with open(pose_path, 'w') as f:
            json.dump(self.hand_pose, f, indent=4)
        print(f"Hand pose estimation results saved to {pose_path}")


                    
if __name__ == "__main__":
    data_replay = HandPoseEstimator("/home/dell/maple_control/data/20250908_fruits_and_tray_hand/20250908_213213")

    data_replay.replay()