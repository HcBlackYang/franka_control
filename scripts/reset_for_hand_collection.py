# -0.28685441 -0.94339818  1.14128171 -1.79782548 -0.09203885  1.14588365 2.66452463

from franky import *


robot = Robot('172.16.0.2')
robot.relative_dynamics_factor = 0.05
robot.recover_from_errors()
gripper = Gripper('172.16.0.2')
gripper.open(0.1)

motion = JointMotion([  1.2171974 , -0.23307747 , 0.01019219 ,-1.4702677 , -0.53225099 , 1.0256551,2.99087099])

robot.move(motion)


state = robot.current_cartesian_state.pose.end_effector_pose.translation
print("Current Cartesian State:", state)