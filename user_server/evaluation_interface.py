from flask import Flask, render_template, request, jsonify
import threading
import time
import os
import json
import numpy as np
import cv2
import sys
import logging
# Add the parent directory to sys.path to import from systems
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from systems.robot_evaluation_system import RobotEvaluationSystem

#####################################################
DATA_SAVE_PATH = "/home/dell/maple_control/data/evaluation/20251014_test_by_asteria"
ACTION_ONLY_MODE = False  # 是否只使用动作空间进行控制
CALIBRATION =True
#####################################################

# 导入您原始脚本中的必要函数和类
TASKS = [
    # "stack the paper cups",
    # "stack the rings on the pillar"
    # "clean up the table",

    "pick up the water bottle",
    "pick up the water plastic bottle",
    "pick up the water paper cup",
    "pick up the paper cup",

    # "pick up the tomato and put it in the yellow tray",
    # "pick up the tomato and put it in the blue tray",

    # "pick up the tomato and put it in the basket",
    # "pick up the pepper and put it in the yellow tray",
    # "pick up the pepper and put it in the blue tray",
    # "pick up the pepper and put it in the basket",
    # "pick up the broccoli and put it in the yellow tray",
    # "pick up the broccoli and put it in the blue tray",
    # "pick up the broccoli and put it in the basket",
]

app = Flask(__name__)

# Global variables for robot system state
robot_system = None
is_running = False
has_calibration = False
current_task = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def initialize_robot_system():
    """Initialize the robot manipulation system"""
    global robot_system, has_calibration
    try:
        if robot_system is None:
            robot_system = RobotEvaluationSystem(save_dir=DATA_SAVE_PATH,action_only_mode=ACTION_ONLY_MODE, calibration=CALIBRATION)
        else:
            robot_system.reset_for_collection()
        has_calibration = True
        logging.info("Robot manipulation system initialized successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize robot system: {e}")
        has_calibration = False
        return False

@app.route('/')
def index():
    """渲染主页面"""
    return render_template('index_eval.html', tasks=TASKS)

@app.route('/status', methods=['GET'])
def get_status():
    """获取当前状态"""
    global is_running, has_calibration
    return jsonify({
        'status': 'success',
        'recording': is_running,
        'calibrating': False,
        'has_calibration': has_calibration
    })

@app.route('/start_recording', methods=['POST'])
def start_recording():
    """开始机器人任务"""
    global robot_system, is_running, current_task
    
    if is_running:
        return jsonify({'status': 'error', 'message': '机器人已在运行中'})
    
    if not has_calibration or robot_system is None:
        return jsonify({'status': 'error', 'message': '请先初始化机器人系统'})
    
    data = request.get_json()
    task = data.get('task', '').strip()
    custom_task = data.get('custom_task', '').strip()
    
    # Use custom task if provided, otherwise use selected task
    task_name = custom_task if custom_task else task
    
    if not task_name:
        return jsonify({'status': 'error', 'message': '请选择或输入任务'})
    
    try:
        current_task = task_name
        robot_system.run(task_name=task_name)
        is_running = True
        logging.info(f"Started robot task: {task_name}")
        return jsonify({
            'status': 'success',
            'message': f'机器人任务已开始: {task_name}',
            'task': task_name
        })
    except Exception as e:
        logging.error(f"Failed to start robot task: {e}")
        return jsonify({'status': 'error', 'message': f'启动失败: {str(e)}'})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    """停止机器人任务（成功完成）"""
    global robot_system, is_running, current_task
    
    if not is_running:
        return jsonify({'status': 'error', 'message': '机器人未在运行'})
    
    try:
        robot_system.stop(success=True)
        is_running = False
        task_name = current_task
        current_task = None
        logging.info(f"Stopped robot task successfully: {task_name}")
        return jsonify({
            'status': 'success',
            'message': f'任务成功完成: {task_name}'
        })
    except Exception as e:
        logging.error(f"Failed to stop robot task: {e}")
        return jsonify({'status': 'error', 'message': f'停止失败: {str(e)}'})

@app.route('/remove_recording', methods=['POST'])
def remove_recording():
    """停止机器人任务（标记为失败）"""
    global robot_system, is_running, current_task
    
    if not is_running:
        return jsonify({'status': 'error', 'message': '机器人未在运行'})
    
    try:
        robot_system.stop(success=False)
        is_running = False
        task_name = current_task
        current_task = None
        logging.info(f"Stopped robot task as failed: {task_name}")
        return jsonify({
            'status': 'success',
            'message': f'任务已标记为失败: {task_name}'
        })
    except Exception as e:
        logging.error(f"Failed to remove robot task: {e}")
        return jsonify({'status': 'error', 'message': f'作废失败: {str(e)}'})

@app.route('/initialize_system', methods=['POST'])
def initialize_system():
    """重置机器人到收集位置"""
    global robot_system, is_running, has_calibration

    if is_running:
        return jsonify({'status': 'error', 'message': '机器人正在运行中，无法重置'})

    success = initialize_robot_system()
    if not success:
        return jsonify({'status': 'error', 'message': '请先初始化机器人系统'})
    else:
        return jsonify({'status': 'success', 'message': '机器人已重置到收集位置'})

@app.route('/start_placer', methods=['POST'])
def start_placer():
    pass

if __name__ == '__main__':
    # Initialize robot system on startup

    app.run(host='0.0.0.0', port=5002, debug=False) # host='0.0.0.0' 允许外部访问