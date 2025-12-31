import os
from pathlib import Path
import json



def cal_states(dir: str = None):
    EVAL_DATA= "eval_info.json"
    task_states = {}
    success_count = 0
    fail_count = 0
    for i, file in enumerate(sorted(os.listdir(dir))):
        # if i+1 in [9, 10, 11, 12, 21, 22, 23, 24, 34, 35, 36, 37, 46, 47, 48, 49]:
        #     print(f"skip:{i, file}")
        #     continue
        eval_data_path = Path(dir) / Path(file) / EVAL_DATA
        with open(eval_data_path, 'r') as f:
            eval_data = json.load(f)
            success = eval_data.get("success", False)
            if success:
                success_count += 1
            else:
                fail_count += 1
    print(f"\nSuccess count: {success_count}, Fail count: {fail_count}")
    print(f"Success rate: {success_count / (success_count + fail_count):.2%}")
if __name__ == "__main__":
    # cal_states("/home/dell/maple_control/data/evaluation/20250911_fps30-10")
    # cal_states("/home/dell/maple_control/data/evaluation/20250915_fps15-10")
    # cal_states("/home/dell/maple_control/data/evaluation/20250912_fps5-10")
    cal_states("/home/dell/maple_control/data/evaluation/20250925_fruits_and_tray_500epi_4gpu_robotandhumandata_baseline_randmaskaction2ego0.0-80k")