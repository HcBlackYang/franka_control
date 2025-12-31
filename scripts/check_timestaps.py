import json
import os 
import pathlib


timestamp_files = "/home/dell/maple_control/data/videotest/top_image_timestamps.json"

if __name__ == "__main__":
    with open(timestamp_files, 'r') as f:
        timestamps = json.load(f)
    last_timestamp = None
    for idx, ts in enumerate(timestamps):
        if last_timestamp is not None:
            diff = ts - last_timestamp
            if diff > 0.1:
                print(f"Large timestamp gap at index {idx}: {diff} seconds")
            print(diff)
        last_timestamp = ts
