import numpy as np
import time
from robots.franky_env import FrankyEnv, ActionSpace

def get_higher_position():
    print("=== 开始校准初始高度 ===")
    
    # 1. 初始化环境，使用 EEF_VELOCITY (末端速度控制) 模式
    # 注意：这会让机械臂先回到目前的旧初始位置
    env = FrankyEnv(action_space=ActionSpace.EEF_VELOCITY)
    
    print("1. 机械臂已回到旧原点")
    time.sleep(1)
    
    print("2. 正在尝试向上抬升 5cm ...")
    # 构造动作: [Vx, Vy, Vz, Wx, Wy, Wz]
    # Z轴速度设为 0.05 m/s (5cm/s)
    action = np.array([0.0, 0.0, 0.07, 0.0, 0.0, 0.0])
    
    # 执行动作: 持续 1000 毫秒 (1秒)
    # 距离 = 速度 * 时间 = 0.05 m/s * 1s = 0.05m = 5cm
    env.step(action, duration=1000)
    
    # 等待一小会儿让机械臂稳定
    time.sleep(0.5)
    
    # 3. 获取当前的关节角度
    new_joints = env.get_position(action_space=ActionSpace.JOINT_ANGLES)
    
    # 格式化输出，方便你直接复制
    joint_str = ", ".join([f"{x:.6f}" for x in new_joints])
    
    print("\n" + "="*50)
    print("✅ 请复制下面的数组，替换 franky_env.py 第13行：")
    print(f"np.array([{joint_str}])")
    print("="*50 + "\n")

    # 停止并释放资源
    env.stop()

if __name__ == "__main__":
    get_higher_position()