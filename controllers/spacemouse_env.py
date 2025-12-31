from controllers.controller_env import ControllerEnv
from robots.franky_env import FrankyEnv
from robots.robot_env import RobotEnv
import numpy as np
from common.constants import ActionSpace
import time
import threading
import logging
import pyspacemouse
import copy 

class SpaceMouseEnv(ControllerEnv):
    def __init__(self, robot_env: RobotEnv):
        super().__init__(robot_env, "SpaceMouseController")
        # Additional initialization for SpaceMouseEnv can be added here
        self._state = {
            "translation": np.zeros(3),  # 位移
            "rotation": np.zeros(3),  # 旋转
            "action": np.zeros(6),  # 综合动作
            "buttons": {
                "left": False,
                "right": False
            },
            "gripper": {
                "target_position": self.robot_env.get_gripper_state(),  # 目标位置 1=关闭, -1=打开
                "current_position": self.robot_env.get_gripper_state(),  # 当前夹爪位置
            },
            "movement_enabled": False,
            "controller_on": False,
            "timestamp": None,
        }
        # 添加线程锁
        self._state_lock = threading.RLock()
    
    def reset(self):
        pass

    def _update_internal_state(self, num_wait_sec=5, hz=200):
        """更新SpaceMouse内部状态"""
        last_read_time = time.time()
        logging.info("🔄 正在启动SpaceMouse线程...")
        try:
            pyspacemouse.open()
        except Exception as e:
            logging.error(f"SpaceMouse打开失败: {e}, 请检查spacemouse是否连接，是否开启权限sudo chmod a+rw /dev/hidraw* ")
            return
        logging.info("🎮 SpaceMouse线程已启动，开始监听设备...")
        
        while not self._stop_controlling.is_set():
            # 调节读取频率
            time.sleep(1 / hz)
            
            # 读取SpaceMouse
            time_since_read = time.time() - last_read_time
            self._state["controller_on"] = time_since_read < num_wait_sec
            # 读取SpaceMouse实际数据
            try:
                # 读取SpaceMouse状态
                state = pyspacemouse.read()
                if state is not None:
                    # 提取位移和旋转数据
                    translation = np.array([state.x, state.y, state.z]) / 1000.0  # 转换为米
                    rotation = np.array([state.roll, state.pitch, state.yaw]) / 1000.0  # 转换为弧度
                    # 提取按钮状态
                    buttons = {
                        "left": state.buttons[0] if len(state.buttons) > 0 else False,
                        "right": state.buttons[1] if len(state.buttons) > 1 else False
                    }
                    with self._state_lock:
                        # 处理夹爪控制逻辑
                        self._state["gripper"]["target_position"] = 1.0 if buttons["left"] else ( -1.0 if buttons["right"] else self._state["gripper"]["target_position"])
                        self._state["translation"] = translation
                        self._state["rotation"] = rotation
                        self._state["action"]  = np.concatenate([np.array([-translation[1],
                                             translation[0],
                                            translation[2]])*100,
                                              np.array([-rotation[0],
                                                        -rotation[1],
                                                  -rotation[2]])*500 
                                              ])
                        self._state["buttons"] = buttons
                        self._state["movement_enabled"] = True
                        self._state["controller_on"] = True
                        self._state["timestamp"] = time.time()
                        last_read_time = time.time()
                else:
                    # 如果没有读取到数据，保持之前的状态
                    pass
                
            except Exception as e:
                print(f"SpaceMouse读取错误: {e}")
                continue
    def _update_robot_state(self, hz=250):
        """更新机器人状态"""
        try:
            while not self._stop_monitoring.is_set():

                gripper_action = 0
                with self._state_lock:
                    if not self._state["movement_enabled"] or not self._state["controller_on"]:
                        logging.warning("Movement is disabled or controller is off. Waiting for SpaceMouse input...")
                        logging.warning("Run sudo chmod a+rw /dev/hidraw* and restart")
                        time.sleep(100 / hz)
                        continue
                    
                    # 获取当前的位移和旋转
                    translation = self._state["translation"]
                    rotation = self._state["rotation"]
                    # 更新夹爪状态
                    if self._state["gripper"]["target_position"] != self._state["gripper"]["current_position"]:
                        if self._state["gripper"]["target_position"] == 1.0:
                            # self.robot_env.open_gripper(asynchronous=True)
                            gripper_action = 1.0
                        else:
                            # self.robot_env.close_gripper(asynchronous=True)
                            gripper_action = -1.0
                        self._state["gripper"]["current_position"] = self._state["gripper"]["target_position"]
                # 动作映射
                if gripper_action == 1.0:
                    logging.info("Opening gripper...")
                    self.robot_env.close_gripper(asynchronous=True)
                elif gripper_action == -1.0:
                    logging.info("Closing gripper...")
                    self.robot_env.open_gripper(asynchronous=True)
                action = np.concatenate([np.array([-translation[1],
                                            translation[0],
                                        translation[2]])*100,
                                            np.array([-rotation[0],
                                                    -rotation[1],
                                                -rotation[2]])*500 
                                            ])
                
                # 执行机器人动作
                self.robot_env.step(action, asynchronous=True)
                    
                    
                time.sleep(1 / hz)
        except Exception as e:
            logging.error(f"Error in _update_robot_state: {e}")
        finally:
            with self._state_lock:
                self._state["movement_enabled"] = False
                self._state["controller_on"] = False
                self.robot_env.stop()
                logging.info("Robot control thread stopped.")
if __name__ == "__main__":
    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),  # 输出到控制台
            ]
        )
    # Example usage
    robot_env = FrankyEnv(action_space=ActionSpace.EEF_VELOCITY)
    env = SpaceMouseEnv(robot_env=robot_env)
    env.start_controlling()

    print(env)
    from cameras.realsense_env import RealSenseEnv
    camera = RealSenseEnv(camera_name="wrist_image", serial_number="342222072092", width=1280, height=720)
    camera.start_monitoring()
    import cv2
    while True:
        main_image = camera.get_latest_frame()
        if main_image is not None and main_image['bgr'] is not None:
            cv2.imshow("Wrist Camera", main_image['bgr'])
            cv2.waitKey(1)
        else:
            logging.warning("No frame received from camera.")
    time.sleep(20000)  # 让线程运行一段时间
    env.stop_monitoring()
    print("Monitoring stopped.")
    pyspacemouse.close()  # 确保关闭SpaceMouse连接
