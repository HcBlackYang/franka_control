# from robots.franky_env import FrankyEnv
# from controllers.gello_env import GelloEnv
# from controllers.spacemouse_env import SpaceMouseEnv
# from cameras.realsense_env import RealSenseEnv
# from cameras.usb_env import USBEnv

# from common.constants import ActionSpace
# import time
# from pathlib import Path
# import logging
# from systems.robot_policy_utils import WebsocketClientPolicy
# from systems.tcp_client import TCPClientPolicy
# import cv2
# import numpy as np
# from cameras.camera_param import CameraParam
# from robots.robot_param import RobotParam
# import math
# import threading

# class RobotPolicySystem:
#     def __init__(self, action_space: ActionSpace = ActionSpace.JOINT_ANGLES, ip: str = "10.21.40.5", port: str = "8003", 
#                  action_only_mode: bool = False, calibration: bool=True):
#         # 初始化机器人环境
#         self.action_space = action_space
#         self.action_only_mode = action_only_mode

#         self.robot_env = FrankyEnv(action_space=action_space, inference_mode=True, robot_param=RobotParam(np.array([ 0.0, 0.0, -math.pi / 2]), np.array([ 0.53433071, 0.52905707, 0.00440881])))
#         if self.action_space not in [ActionSpace.EEF_VELOCITY, ActionSpace.JOINT_ANGLES]:
#             raise NotImplementedError(f"Action space '{self.action_space}' is not supported.")
#         logging.info(f"Trying to connect to policy server at {ip}:{port}...")
#         # self.client = WebsocketClientPolicy(
#         #     host= ip,
#         #     port= port
#         # )
#         self.client = TCPClientPolicy(
#             host= ip,
#             port= port
#         )
#         logging.info(f"Connected to policy server at {ip}:{port}.")
        
#         self.main_camera = RealSenseEnv(camera_name="main_image", serial_number="339322073638", width=1280, height=720,
#                                         camera_param=CameraParam(intrinsic_matrix = np.array([[908.1308, 0, 655.7268], [0, 910.0818, 395.8856], [0, 0, 1]], dtype=np.float32),
#                                                                  distortion_coeffs = np.array([0.1068, -0.2123, -0.0092, 0.0000, 0.0000], dtype=np.float32)))
#         self.wrist_camera = RealSenseEnv(camera_name="wrist_image", serial_number="342222072092", width=1280, height=720)
        
#         # 只在非action_only模式下初始化top_camera
        
#         self.top_camera = USBEnv(camera_name="top_image", serial_number="12", width=1920, height=1080, exposure=100,
#                         camera_param=CameraParam(np.array([[1158.0, 0, 999.9484], [0, 1159.9, 584.2338], [0, 0, 1]], dtype=np.float32), np.array([0.0412, -0.0509, 0.0000, 0.0000, 0.0000], dtype=np.float32))
#                     )
#         if calibration:
#             self.main_camera.calib_camera()
#             self.top_camera.calib_camera()

#         self.gripper_status = {
#             "current_state": 0,
#             "target_state": 0 
#         }
#         self.stop_evaluation = threading.Event()
#         self.all_action_and_traj = []
#         self.all_action_and_traj_lock = threading.Lock()

#     def reset_for_collection(self):
#         """重置机器人到随机位置，用于数据收集"""
#         self.robot_env.reset()
#         action = np.array([0,0,-0.05,0,0,0])
#         self.robot_env.step(action, asynchronous=False)
#         return True



#     def run(self, show_image: bool = False, task_name: str = "default_task"):
#         self.main_camera.start_monitoring()
#         self.wrist_camera.start_monitoring()
#         self.top_camera.start_monitoring()
#         # self.robot_env.step(np.array([0.01,0.01, -0.02,0,0,0]), asynchronous=False)

#         self.gripper_status = {
#             "current_state": 0,
#             "target_state": 0 
#         }
#         self.stop_evaluation.clear()
#         all_action_and_traj = []
#         while not self.stop_evaluation.is_set():
#             main_image = self.main_camera.get_latest_frame()['bgr']
#             wrist_image = self.wrist_camera.get_latest_frame()['bgr']
#             top_image = self.top_camera.get_latest_frame()['bgr']

#             if main_image is None or wrist_image is None:
#                 time.sleep(0.05)
#                 continue
                
#             joint_angles = self.robot_env.get_position(action_space=ActionSpace.JOINT_ANGLES)
#             gripper_width = self.robot_env.get_gripper_width()
#             eef_pose = self.robot_env.get_position(action_space=ActionSpace.EEF_POSE)
#             state = np.concatenate([eef_pose, [gripper_width]])
#             # 根据模式选择不同的处理逻辑
#             if self.action_only_mode:
#                 state_trajectory = eef_pose[:3]
#                 element = {
#                     "observation/image": main_image,
#                     "observation/wrist_image": wrist_image,
#                     "observation/state": state,
#                     "prompt": task_name,
#                 }
#             else:
#                 state_trajectory = self.robot_env.robot_param.transform_to_world(np.array([eef_pose[:3]]))[0]
#                 element = {
#                     "observation/image": main_image,
#                     "observation/wrist_image": wrist_image,
#                     "observation/state": state,
#                     "qpos": joint_angles.tolist(),
#                     "observation/state_trajectory": state_trajectory,
#                     "prompt": task_name,
#                 }

#             inference_results = self.client.infer(element)
#             actions_chunk = np.array(inference_results["actions"])
            
#             if not self.action_only_mode:
#                 trajectory_chunk = np.array(inference_results["trajectory"])
#             all_action_and_traj.append({
#                 'actions': actions_chunk.tolist(),
#                 'trajectory': trajectory_chunk.tolist() if not self.action_only_mode else None,
#                 'timestamp': time.time(),
#                 'state': state.tolist(),
#                 'state_trajectory': state_trajectory.tolist() if not self.action_only_mode else None
#             }.copy())
#             with self.all_action_and_traj_lock:
#                 self.all_action_and_traj = all_action_and_traj

#             cnt = 0
            
#             if show_image:
#                 draw_main_image = main_image.copy()
                
                
#                 draw_top_image = top_image.copy()
                
#                 action_trajectory = 0.1 * np.cumsum(actions_chunk,axis=0)
#                 action_trajectory_in_world = self.robot_env.robot_param.transform_to_world(action_trajectory[:,:3] + eef_pose[:3])
#                 if not self.action_only_mode:
#                     draw_main_image = self.main_camera.camera_param.draw_trajectory_on_image(draw_main_image, trajectory_chunk)
#                     draw_top_image = self.top_camera.camera_param.draw_trajectory_on_image(draw_top_image, trajectory_chunk)

#                 draw_main_image = self.main_camera.camera_param.draw_trajectory_on_image(draw_main_image, action_trajectory_in_world)
#                 draw_top_image = self.top_camera.camera_param.draw_trajectory_on_image(draw_top_image, action_trajectory_in_world)
#                 cv2.imshow("Top Camera", draw_top_image)
                
#                 cv2.imshow("Main Camera", draw_main_image)
#                 cv2.imshow("Wrist Camera", wrist_image)
#                 cv2.waitKey(1)

#             for action in actions_chunk:
#                 self.robot_env.step(action[:-1], asynchronous=True)
#                 time.sleep(0.1)

#                 cnt += 1
#                 gripper_action = action[-1]
                

#                 if gripper_action > 0.95:
#                     self.gripper_status["target_state"] = 1
#                 elif gripper_action < -0.95:
#                     self.gripper_status["target_state"] = -1
                
#                 if self.gripper_status["current_state"] != self.gripper_status["target_state"]:
#                     if self.gripper_status["target_state"] == -1:
#                         self.robot_env.open_gripper(asynchronous=True)
#                     else:
#                         self.robot_env.close_gripper(asynchronous=True)
#                     self.gripper_status["current_state"] = self.gripper_status["target_state"]
                
#                 max_cnt = 10

#                 if cnt == max_cnt:
#                     self.robot_env.step(np.array([0,0,0,0,0,0]), asynchronous=False)
#                     break
#     def stop(self):
#         self.stop_evaluation.set()
#         time.sleep(0.5)

#         self.robot_env.stop_saving_state()
#         logging.info("Robot policy system stopped.")

# # if __name__ == "__main__":
# #     logging.basicConfig(
# #             level=logging.INFO,
# #             format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
# #             handlers=[
# #                 logging.StreamHandler(),  # 输出到控制台
# #             ]
# #         )
# #     # 使用 action_only_mode 参数控制模式
# #     # action_only_mode=True 对应原来的 robot_policy_action_only_system
# #     # action_only_mode=False 对应原来的 robot_policy_system
# #     system = RobotPolicySystem(action_space=ActionSpace.EEF_VELOCITY, action_only_mode=True, prompt="pick up the water bottle",
# #                               camera_calib_file="/home/dell/maple_control/data/20250829_fruits_and_tray/20250829_173310")
# #     system.run(show_image=True)

# if __name__ == "__main__":
#     logging.basicConfig(
#             level=logging.INFO,
#             format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#             handlers=[
#                 logging.StreamHandler(),
#             ]
#         )
    
#     # === 修改点 1: 初始化参数修正 ===
#     # 1. action_space 改为 JOINT_ANGLES (因为 RDT 输出的是 7 维关节角)
#     # 2. 去掉 prompt 和 camera_calib_file (原类定义中没有这些参数)
#     # 3. ip 改为你的服务器 IP (如果是同一台机器用 127.0.0.1)
#     system = RobotPolicySystem(
#         action_space=ActionSpace.JOINT_ANGLES, 
#         ip="127.0.0.1", 
#         port=6000,
#         action_only_mode=False
#     )
    
#     # === 修改点 2: 在 run 中传入指令 ===
#     # task_name 对应原来的 prompt
#     system.run(show_image=True, task_name="pick up the water bottle")


# import sys
# import os
# from robots.franky_env import FrankyEnv
# from common.constants import ActionSpace
# import time
# import logging
# from systems.tcp_client import TCPClientPolicy 
# import cv2
# import numpy as np
# from robots.robot_param import RobotParam
# import math
# import threading

# class RobotPolicySystem:
#     def __init__(self, action_space: ActionSpace = ActionSpace.JOINT_ANGLES, ip: str = "127.0.0.1", port: int = 6000, 
#                  action_only_mode: bool = False, calibration: bool=True):
#         self.action_space = action_space
#         self.action_only_mode = action_only_mode

#         # 初始化 Franka 机器人
#         # inference_mode=True 通常意味着更灵敏的控制响应
#         self.robot_env = FrankyEnv(
#             action_space=action_space, 
#             inference_mode=True, 
#             robot_param=RobotParam(np.array([ 0.0, 0.0, -math.pi / 2]), np.array([ 0.53433071, 0.52905707, 0.00440881]))
#         )
        
#         logging.info(f"Trying to connect to policy server at {ip}:{port}...")
#         self.client = TCPClientPolicy(host=ip, port=port)
#         logging.info(f"Connected to policy server at {ip}:{port}.")
        
#         # Camera 初始化
#         from cameras.realsense_env import RealSenseEnv
        
#         # 只启动手腕相机 (Wrist-Only Inference)
#         self.wrist_camera = RealSenseEnv(camera_name="wrist_image", serial_number="342222072092", width=1280, height=720)
#         self.gripper_status = {"current_state": 0, "target_state": 0}
#         self.stop_evaluation = threading.Event()

#     def run(self, show_image: bool = False, task_name: str = "default_task"):
#         self.wrist_camera.start_monitoring()
        
#         logging.info("Waiting 2.0s for cameras to warm up...")
#         time.sleep(2.0)
        
#         logging.info("Starting inference loop...")
        
#         # =========================================================
#         # 🔧 核心参数调优
#         # 1. EXECUTION_HORIZON: 设为 64 (与模型预测长度一致)
#         #    这能彻底消除在目标附近的"犹豫"和"反复横跳"。
#         # 2. CONTROL_FREQUENCY: 25Hz (每步 0.04s)
#         # 3. MAX_STEP_RAD: 关节动作限幅，防止剧烈抖动
#         # =========================================================
#         EXECUTION_HORIZON = 64
#         CONTROL_FREQUENCY = 25  
#         STEP_DURATION = 1.0 / CONTROL_FREQUENCY # 0.04s
        
#         # 0.05 弧度 ≈ 2.8度。限制每 0.04s 最多转这么大角度，防止抽搐。
#         MAX_STEP_RAD = 0.05 
        
#         last_executed_joints = None

#         while not self.stop_evaluation.is_set():
#             t0 = time.time()
            
#             # 1. 获取图像
#             wrist_frame_data = self.wrist_camera.get_latest_frame()
#             if wrist_frame_data is None:
#                 time.sleep(0.01)
#                 continue
            
#             wrist_image = wrist_frame_data['bgr']
#             # 构造全黑主摄占位符 (适配训练时的 Modality Dropout)
#             main_image = np.zeros_like(wrist_image)

#             # 2. 获取状态
#             joint_angles = self.robot_env.get_position(action_space=ActionSpace.JOINT_ANGLES)
#             gripper_width = self.robot_env.get_gripper_width()
#             eef_pose = self.robot_env.get_position(action_space=ActionSpace.EEF_POSE)
            
#             # 构造 8 维 qpos (7关节 + 1夹爪)
#             qpos_8d = list(joint_angles) + [float(gripper_width)]
#             # 构造 State
#             state = np.concatenate([eef_pose, [gripper_width]])
            
#             # 3. 构造请求
#             element = {
#                 "observation/agentview_image": main_image, 
#                 "observation/wrist_image": wrist_image,
#                 "observation/state": state,
#                 "qpos": qpos_8d, 
#                 "prompt": task_name,
#             }

#             # 4. 推理 (Blocking)
#             inference_results = self.client.infer(element)
            
#             if inference_results and "actions" in inference_results:
#                 new_actions = inference_results["actions"][0] # [64, 8]
                
#                 # 健壮性检查
#                 if not isinstance(new_actions, list) or len(new_actions) == 0:
#                     continue

#                 # 截取要执行的片段 (全量执行以消除犹豫)
#                 actions_to_execute = new_actions[:EXECUTION_HORIZON]
                
#                 print(f"  >>> Executing chunk ({len(actions_to_execute)} steps)...")

#                 for i, action in enumerate(actions_to_execute):
#                     # 类型检查
#                     if not isinstance(action, (list, tuple, np.ndarray)):
#                         continue

#                     # [关键 1] 强制转为 float64，满足 C++ 接口要求
#                     action_np = np.array(action, dtype=np.float64)

#                     # [关键 2] 空动作/非法值拦截
#                     if np.all(action_np == 0) or np.isnan(action_np).any():
#                         print(f"\r⚠️ Invalid action detected, skipping chunk.", end="")
#                         break 
                    
#                     target_joints = action_np[:-1] # 前7位 (Joints)
#                     gripper_val = action_np[-1]    # 第8位 (Gripper)

#                     # [关键 3] 平滑限幅逻辑 (Anti-Jitter)
#                     if last_executed_joints is not None:
#                         # 计算差值
#                         diff = target_joints - last_executed_joints
#                         # 限制最大变化量 (Clip)
#                         diff_clipped = np.clip(diff, -MAX_STEP_RAD, MAX_STEP_RAD)
#                         # 应用限制后的新目标
#                         target_joints = last_executed_joints + diff_clipped
                    
#                     # 更新记录
#                     last_executed_joints = target_joints.copy()

#                     t_step_start = time.time()
                    
#                     # A. 关节控制 (异步执行)
#                     self.robot_env.step(target_joints, asynchronous=True)
                    
#                     # B. 夹爪控制 (带状态机)
#                     if gripper_val > 0.06: 
#                          if self.gripper_status["current_state"] != -1:
#                              self.robot_env.open_gripper(asynchronous=True)
#                              self.gripper_status["current_state"] = -1
#                     elif gripper_val < 0.02:
#                          if self.gripper_status["current_state"] != 1:
#                              self.robot_env.close_gripper(asynchronous=True)
#                              self.gripper_status["current_state"] = 1
                    
#                     # C. 频率控制 (25Hz)
#                     dt = time.time() - t_step_start
#                     remain = STEP_DURATION - dt
#                     if remain > 0: 
#                         time.sleep(remain)

#             # 可视化 (仅在推理间隙刷新，避免阻塞控制循环)
#             if show_image:
#                 cv2.imshow("Wrist View", wrist_image)
#                 cv2.waitKey(1)

#             latency = (time.time() - t0) * 1000
#             print(f"\rChunk Latency: {latency:.1f}ms", end="")

#     def stop(self):
#         self.stop_evaluation.set()
#         time.sleep(0.5)
#         logging.info("System stopped.")

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     system = RobotPolicySystem(
#         action_space=ActionSpace.JOINT_ANGLES, 
#         ip="127.0.0.1", 
#         port=6000
#     )
#     try:
#         system.run(show_image=True, task_name="pick up the paper cup")
#     except KeyboardInterrupt:
#         system.stop()

import sys
import os
import time
import logging
import cv2
import numpy as np
import math
import threading
from collections import deque
from common.constants import ActionSpace
from robots.franky_env import FrankyEnv
from robots.robot_param import RobotParam
from systems.tcp_client import TCPClientPolicy 

# 引入你的相机库
from cameras.realsense_env import RealSenseEnv

class ImageRecorder(threading.Thread):
    def __init__(self, camera, buffer_size=16):
        super().__init__()
        self.camera = camera
        self.buffer_size = buffer_size
        self.running = False
        self.lock = threading.Lock()
        
        # 两个 Buffer：
        # 1. raw_buffer: 存原始图，用于显示
        # 2. video_buffer: 存处理后的 tensor/numpy，用于推理
        self.latest_frame = None
        
        # 这里的 buffer 只要存 numpy 数组即可，不需要存 Tensor，
        # 转换 Tensor 的工作交给 Server 端，或者在发送前做，减少传输压力
        # 但为了配合你的 Server 逻辑，我们这里只存原始 BGR 图像
        self.frame_buffer = deque(maxlen=buffer_size) 
        self.stop_event = threading.Event()

    def run(self):
        self.running = True
        self.camera.start_monitoring()
        logging.info("[ImageRecorder] Background thread started.")
        
        while not self.stop_event.is_set():
            # 获取最新帧 (这是轻量级操作)
            data = self.camera.get_latest_frame()
            if data is not None:
                img = data['bgr']
                
                with self.lock:
                    self.latest_frame = img.copy()
                    # 存入 Buffer
                    self.frame_buffer.append(img)
                
                # 实时显示 (在这里显示最流畅)
                cv2.imshow("Wrist View (Real-time)", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()
            
            # 保持约 30Hz 的采样率 (根据你训练数据的帧率调整)
            # 如果你训练是 10Hz，这里改成 time.sleep(0.1)
            time.sleep(0.033) 
        
        cv2.destroyAllWindows()
        logging.info("[ImageRecorder] Stopped.")

    def get_inference_input(self):
        """
        获取用于推理的 snapshot。
        如果 Buffer 还没满，就用第一帧复制填充 (Padding)。
        """
        with self.lock:
            if len(self.frame_buffer) == 0:
                return None, None
            
            current_img = self.latest_frame.copy()
            
            # 拿到 Buffer 的快照
            frames_snapshot = list(self.frame_buffer)
        
        # 策略：如果不够 16 帧，用第一帧补齐头部 (Padding Head)
        # 这样保证时序相对关系是正确的
        while len(frames_snapshot) < self.buffer_size:
            frames_snapshot.insert(0, frames_snapshot[0])
            
        
        return current_img

    def stop(self):
        self.stop_event.set()
        self.join()

class RobotPolicySystem:
    def __init__(self, action_space: ActionSpace = ActionSpace.JOINT_ANGLES, ip: str = "127.0.0.1", port: int = 6000):
        self.action_space = action_space
        
        # Robot
        self.robot_env = FrankyEnv(
            action_space=action_space, 
            inference_mode=True, 
            robot_param=RobotParam(np.array([ 0.0, 0.0, -math.pi / 2]), np.array([ 0.53433071, 0.52905707, 0.00440881]))
        )
        
        # Client
        logging.info(f"Connecting to {ip}:{port}...")
        self.client = TCPClientPolicy(host=ip, port=port)
        logging.info("Connected.")
        
        # Camera & Recorder
        self.wrist_camera = RealSenseEnv(camera_name="wrist_image", serial_number="342222072092", width=1280, height=720)
        # 启动后台采集线程
        self.recorder = ImageRecorder(self.wrist_camera, buffer_size=16)
        
        self.gripper_status = {"current_state": 0}
        self.stop_evaluation = threading.Event()

    def run(self, task_name: str = "default_task"):
        # 启动后台采集
        self.recorder.start()
        
        logging.info("Waiting 2.0s for warmup...")
        time.sleep(2.0)
        
        # 参数设置
        EXECUTION_HORIZON = 15  # 信任模型，做完 15 步
        MAX_STEP_RAD = 0.05     # 限幅
        last_executed_joints = None
        
        logging.info("Starting inference loop...")

        try:
            while not self.stop_evaluation.is_set():
                if not self.recorder.is_alive():
                    break

                t0 = time.time()
                
                # 1. 从后台线程拿【最新鲜】的一张图
                # 即使主线程卡了 5 秒，这里拿到的也是 0.001 秒前相机刚拍到的
                wrist_image = self.recorder.get_inference_input()
                
                if wrist_image is None:
                    time.sleep(0.01)
                    continue

                # 2. 获取机器人状态
                joint_angles = self.robot_env.get_position(action_space=ActionSpace.JOINT_ANGLES)
                gripper_width = self.robot_env.get_gripper_width()
                eef_pose = self.robot_env.get_position(action_space=ActionSpace.EEF_POSE)
                
                qpos_8d = list(joint_angles) + [float(gripper_width)]
                state = np.concatenate([eef_pose, [gripper_width]])
                
                # 3. 发送请求
                # 注意：我们在 Server 端已经改成了 "收到一张图 -> 复制填满 Buffer" 的静态图策略
                # 这配合这里 "获取最新鲜的一张图" 是目前最稳健的组合
                element = {
                    "observation/agentview_image": np.zeros_like(wrist_image), 
                    "observation/wrist_image": wrist_image,
                    "observation/state": state,
                    "qpos": qpos_8d, 
                    "prompt": task_name,
                }

                # 4. 推理 (Blocking 2.5s)
                inference_results = self.client.infer(element)
                
                if inference_results and "actions" in inference_results:
                    new_actions = inference_results["actions"][0]
                    
                    if not isinstance(new_actions, list) or len(new_actions) == 0:
                        continue

                    # 执行 15 步
                    actions_to_execute = new_actions[:EXECUTION_HORIZON]
                    
                    print(f"  >>> Executing chunk ({len(actions_to_execute)} steps)...")

                    for action in actions_to_execute:
                        if not isinstance(action, (list, tuple, np.ndarray)): continue
                        
                        # 数据处理
                        action_np = np.array(action, dtype=np.float64)
                        if np.all(action_np == 0) or np.isnan(action_np).any(): break
                        
                        target_joints = action_np[:-1]
                        gripper_val = action_np[-1]

                        # 平滑限幅
                        if last_executed_joints is not None:
                            diff = np.clip(target_joints - last_executed_joints, -MAX_STEP_RAD, MAX_STEP_RAD)
                            target_joints = last_executed_joints + diff
                        
                        last_executed_joints = target_joints.copy()

                        # 执行
                        t_step_start = time.time()
                        self.robot_env.step(target_joints, asynchronous=True)
                        
                        # 夹爪
                        if gripper_val > 0.06 and self.gripper_status["current_state"] != -1:
                             self.robot_env.open_gripper(asynchronous=True)
                             self.gripper_status["current_state"] = -1
                        elif gripper_val < 0.02 and self.gripper_status["current_state"] != 1:
                             self.robot_env.close_gripper(asynchronous=True)
                             self.gripper_status["current_state"] = 1
                        
                        # 控频 25Hz
                        remain = 0.04 - (time.time() - t_step_start)
                        if remain > 0: time.sleep(remain)

                latency = (time.time() - t0) * 1000
                print(f"\rLoop Latency: {latency:.1f}ms", end="")

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        self.stop_evaluation.set()
        self.recorder.stop()
        time.sleep(0.5)
        logging.info("System stopped.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    system = RobotPolicySystem(ip="127.0.0.1", port=6000)
    system.run(task_name="pick up the paper cup")