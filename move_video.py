#%%
import os
import shutil
import datetime

def parse_time_from_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if "Start date :" in line:
                # Correctly split the line to extract the full date and time
                date_time_str = line.split('Start date :')[1].strip()
                return datetime.datetime.strptime(date_time_str, '%Y/%m/%d %H:%M:%S')
    return None

def find_and_copy_video(txt_file_path, video_folder_path):
    start_time = parse_time_from_file(txt_file_path)
    if start_time is None:
        print("Start time not found in file.")
        return

    # Calculate the earliest acceptable time (3 minutes before the start time)
    earliest_time = start_time - datetime.timedelta(minutes=3)
    matched_video = None

    for file_name in os.listdir(video_folder_path):
        if file_name.endswith('_output.avi'):
            # Extract time from video file name
            video_time_str = file_name.split('_')[0]
            video_time = datetime.datetime.strptime(video_time_str, '%Y%m%d-%H%M%S')

            # Check if the video time is within the desired range
            if earliest_time <= video_time < start_time:
                matched_video = file_name
                break

    if matched_video:
        shutil.copy(os.path.join(video_folder_path, matched_video), os.path.dirname(txt_file_path))
        print(f"Copied {matched_video} to {os.path.dirname(txt_file_path)}")
    else:
        print("No matching video file found.")

#%%
txt = r"F:\My Drive\0-Main\stress_p\pre\wt\1p-2023-11-06-134807.txt"
video_folder_path = r"N:\users\soheibm\oct_as\post videos"
find_and_copy_video(txt, video_folder_path)
# %%
