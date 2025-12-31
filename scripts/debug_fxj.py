import os
import json

dir = "/home/dell/maple_control/data/evaluation/20250910_friut_and_tray_baseline"

all_time = []
all_time_fail = []
for case in os.listdir(dir):
    with open(os.path.join(dir, case, "eval_info.json"), "r") as f:
        eval_info = json.load(f)
    
    if eval_info['success']:
        all_time.append(eval_info['end_time'] - eval_info['start_time'])
    else:
        all_time_fail.append(eval_info['end_time'] - eval_info['start_time'])

print(sorted(all_time))
print(f"Average time for successful tasks: {sum(all_time)/len(all_time)} seconds")


import matplotlib.pyplot as plt
plt.hist(all_time, bins=20)
plt.hist(all_time_fail, bins=20, alpha=0.5, color='r')
plt.xlabel('Time (seconds)')
plt.ylabel('Number of Tasks')
plt.title('Distribution of Task Completion Times')
plt.show()
