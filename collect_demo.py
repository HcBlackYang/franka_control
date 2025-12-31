from robots.franky_env import FrankyEnv
# from controllers.gello_controller import GelloEnv
from controllers.spacemouse_env import SpaceMouseEnv
from cameras.realsense_env import RealSenseEnv
from common.constants import ActionSpace
import time
from pathlib import Path
if __name__ == "__main__":
    # 初始化机器人环境
    robot_env = FrankyEnv(action_space=ActionSpace.EEF_VELOCITY)
    controller_env = SpaceMouseEnv(robot_env=robot_env)
    save_path = Path("./data/") / time.strftime("%Y%m%d_%H%M%S")
    save_path.mkdir(parents=True, exist_ok=True)
    camera_env = RealSenseEnv(camera_name="main_image", serial_number="339322073638", width=640, height=480)
    # 启动相机监控和保存
    camera_env.start_monitoring()
    controller_env.start_controlling()
    robot_env.start_saving_state(str(save_path))
    controller_env.start_saving_state(str(save_path))
    camera_env.start_saving_frames(str(save_path))
    time.sleep(5)
    # 停止所有保存和监控
    robot_env.stop_saving_state()
    controller_env.stop_saving_state()
    controller_env.stop_controlling()
    camera_env.stop_saving_frames()
    camera_env.stop_monitoring()
    print(f"Data saved to {save_path}")