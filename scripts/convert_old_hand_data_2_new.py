import os
import json
from pathlib import Path
import cv2
TASK_INFO = "task_info.json"
VIDEOS_NAME = [
    "wrist_camera",
    "top_camera",
    "main_camera",
    "human_camera",
]
CALIB_DATA = "calibration_data.json"
TIMESTAMPS_NAME = [f"{name}_video_timestamps.json" for name in VIDEOS_NAME]
POSE_NAME = "hand_mano_pose_estimation_results.json"
MANO_NAME = "mano_results.json"
# ├── calibration_data.json
# ├── hand_mano_pose_estimation_results.json
# ├── human_camera_video.avi
# ├── human_camera_video_timestamps.json
# ├── main_camera_output.avi
# ├── main_camera_video.avi
# ├── main_camera_video_detections.json
# ├── main_camera_video_timestamps.json
# ├── mano_results.json
# ├── task_info.json
# ├── top_camera_output.avi
# ├── top_camera_video.avi
# ├── top_camera_video_detections.json
# ├── top_camera_video_timestamps.json
# ├── wrist_camera_output.avi
# ├── wrist_camera_video.avi
# ├── wrist_camera_video_detections.json
# └── wrist_camera_video_timestamps.json

def convert_old_hand_data_2_new(old_data_path: str):
    # with open(str(Path(old_data_path) / TASK_INFO), 'r') as f:
    #     task_info = json.load(f)
    # print(task_info)
    # new_task_info = {
    #     "name": task_info["task_name"],
    #     "start_time": task_info["date"],
    #     "end_time": task_info["date"],
    #     "success": True,
    # }
    # with open(str(Path(old_data_path) / TASK_INFO), 'w') as f:
    #     json.dump(new_task_info, f, indent=4)
    # convert avi to mp4
    if os.path.exists(str(Path(old_data_path) / "task_info.json")):
        print(f"{old_data_path} already converted, skip")
        return
    for video_name in VIDEOS_NAME:
        avi_path = str(Path(old_data_path) / f"{video_name}_video.avi")
        mp4_path = str(Path(old_data_path) / f"{video_name.replace('camera','image')}.mp4")
        if not os.path.exists(avi_path):
            print(f"{avi_path} not exists, skip")
            continue
        if os.path.exists(mp4_path):
            print(f"{mp4_path} already exists, skip")
            continue
        cap = cv2.VideoCapture(avi_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()
        out.release()
        os.remove(avi_path)
    if os.path.exists(str(Path(old_data_path) / CALIB_DATA)) is False:
        print(f"{str(Path(old_data_path) / CALIB_DATA)} not exists, skip")
        return
    with open(str(Path(old_data_path) / CALIB_DATA), 'r') as f:
        calib_data = json.load(f)
    for cam in VIDEOS_NAME:
        if cam in calib_data:
            cam_calib_data = {
                "intrinsic_matrix": calib_data[cam]["camera_matrix"],
                "distortion_coeffs": calib_data[cam]["dist_coeffs"],
                "extrinsic_tvec": calib_data[cam]["tvec"],
                "extrinsic_rvec": calib_data[cam]["rvec"],
            }
            save_path = str(Path(old_data_path) / f"{cam.replace('camera','image')}_camera_param.json")
            with open(save_path, 'w') as f:
                json.dump(cam_calib_data, f, indent=4)
    
    for cam in VIDEOS_NAME:
        old_timestamps_path = str(Path(old_data_path) / f"{cam}_video_timestamps.json")
        new_timestamps_path = str(Path(old_data_path) / f"{cam.replace('camera','image')}_timestamps.json")
        if not os.path.exists(old_timestamps_path):
            print(f"{old_timestamps_path} not exists, skip")
            continue
        if os.path.exists(new_timestamps_path):
            print(f"{new_timestamps_path} already exists, skip")
            continue
        with open(old_timestamps_path, 'r') as f:
            timestamps = json.load(f)
        with open(new_timestamps_path, 'w') as f:
            json.dump(timestamps, f, indent=4)
    for cam in VIDEOS_NAME:
        old_detections_path = str(Path(old_data_path) / f"{cam}_video_detections.json")
        new_detections_path = str(Path(old_data_path) / f"{cam.replace('camera','image')}_detections.json")
        if not os.path.exists(old_detections_path):
            print(f"{old_detections_path} not exists, skip")
            continue
        if os.path.exists(new_detections_path):
            print(f"{new_detections_path} already exists, skip")
            continue
        with open(old_detections_path, 'r') as f:
            detections = json.load(f)
        with open(new_detections_path, 'w') as f:
            json.dump(detections, f, indent=4)
    
    with open(str(Path(old_data_path) / POSE_NAME), 'r') as f:
        pose_data = json.load(f)
    if not os.path.exists(str(Path(old_data_path) / MANO_NAME)):
        new_pose_data = {
            "joints_3d_batch": pose_data,
        }
    else:
        with open(str(Path(old_data_path) / MANO_NAME), 'r') as f:
            mano_data = json.load(f)
        new_pose_data = {
            "joints_3d_batch": pose_data,
            "betas": mano_data["betas"],
            "global_orient": mano_data["global_orient"],
            "hand_pose": mano_data["hand_pose"],
            "transl": mano_data["transl"],
            "loss": mano_data.get("loss", None)
        }
    with open(str(Path(old_data_path) / "hand_mano_pose_estimation_results.json"), 'w') as f:
        json.dump(new_pose_data, f, indent=4)
    task_info = {
        "name": calib_data.get("task_name", "unknown_task"),
        "success": True,
    }
    with open(str(Path(old_data_path) / TASK_INFO), 'w') as f:
        json.dump(task_info, f, indent=4)

if __name__ == "__main__":
    # conver folders in /home/dell/maple_control/data/stack_rings_hand
    data_root = "/home/dell/maple_control/data/20250729_fruits_and_tray"
    for folder in os.listdir(data_root):
        folder_path = os.path.join(data_root, folder)
        if os.path.isdir(folder_path):
            print(f"Converting {folder_path}")
            convert_old_hand_data_2_new(folder_path)
            print(f"Finished converting {folder_path}")