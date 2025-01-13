import os
from typing import List

import cv2

def export_video(frames : List, video_folder : str, filename : str = "video.mp4", fps : int =30) -> None:
    """
    Write frames to a video file

    :param frames: list of frames to write to video
    :param video_folder: target folder to save video
    :param filename: filename of video (must be .mp4)
    :param fps: frames per second
    :return:
    """
    assert filename.endswith(".mp4"), "Target file must be a .mp4 file"

    os.makedirs(video_folder, exist_ok=True)
    target_file = os.path.join(video_folder, filename)

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(target_file, fourcc, fps, (width, height))
    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video.release()

    print(f"Video saved at {target_file}!")