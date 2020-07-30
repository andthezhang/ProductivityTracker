import os
import glob
from collections import defaultdict

import cv2

DATA_DIR = "data"
ANNOTATION_TRAIN_DIR = os.path.join(DATA_DIR, "annotations/train")
AROUSAL_TRAIN_DIR = os.path.join(DATA_DIR, "annotations/train/arousal")
VALENCE_TRAIN_DIR = os.path.join(DATA_DIR, "annotations/train/valence")
AROUSAL_TRAIN_FILES = os.path.join(AROUSAL_TRAIN_DIR, "*.txt")
VALENCE_TRAIN_FILES = os.path.join(VALENCE_TRAIN_DIR, "*.txt")

BBOXES_TRAIN_DIR = os.path.join(DATA_DIR, "bboxs/train")
LANDMARKS_TRAIN_DIR = os.path.join(DATA_DIR, "landmarks/train")
VIDEOS_TRAIN_DIR = os.path.join(DATA_DIR, "videos/train")
VIDEOS_TEST_DIR = os.path.join(DATA_DIR, "videos/test")

BORED = (-0.34, -0.78)
DROOPY = (-0.32, -0.96)
TIRED = (-0.02, -0.99)
SLEEPY = (0.02, -0.99)
EMOTIONS = [BORED, DROOPY, TIRED, SLEEPY]

def is_bored(valence: float, 
             arousal: float, 
             radius: float=0.1) -> bool:
    """
    Defined bored in valence, arousal. See Fig. 1 in https://arxiv.org/pdf/1804.10938.pdf.
    """
    is_bored = False
    for (emotion_valence, emotion_arousal) in EMOTIONS:
        with_in_radius = (emotion_valence - radius <= valence <= emotion_valence + radius and 
            emotion_arousal - radius <= arousal <= emotion_arousal + radius)
        is_bored = is_bored or with_in_radius
    return is_bored

def get_video_path(video_id: str):
    mp4_path = os.path.join(VIDEOS_TRAIN_DIR, video_id + ".mp4")
    avi_path = os.path.join(VIDEOS_TRAIN_DIR, video_id + ".avi")
    if os.path.isfile(mp4_path):
        return mp4_path
    if os.path.isfile(avi_path):
        return avi_path
    raise ValueError("Video {} does not exist".format(video_id))

def create_candidates(output_dir="candidates", radius: float=0.1) -> str:
    candidate_frames = defaultdict(set)
    total_count = 0
    bored_count = 0
    for arousal_file, valence_file in zip(sorted(glob.glob(AROUSAL_TRAIN_FILES)), sorted(glob.glob(VALENCE_TRAIN_FILES))):
        video_id = os.path.splitext(os.path.basename(arousal_file))[0] # eg. file_id = 105
        arousal_file = open(arousal_file, 'r')
        valence_file = open(valence_file, 'r')
        arousal_values = [float(line) for line in arousal_file.readlines()]
        valence_values = [float(line) for line in valence_file.readlines()]
        for frame_id, (valence_value, arousal_value) in enumerate(zip(valence_values, arousal_values)):
            total_count += 1
            if is_bored(valence_value, arousal_value, radius=radius):
                bored_count += 1
                candidate_frames[video_id].add(frame_id)

    # Store bored frames in "bored" directory, non-bored frames in "non-bored" directory.
    output_dir_bored = os.path.join(output_dir, str(radius), "bored")
    output_dir_non_bored = os.path.join(output_dir, str(radius), "non_bored")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_bored, exist_ok=True)
    os.makedirs(output_dir_non_bored, exist_ok=True)

    for video_id, candidate_frames in candidate_frames.items():
        video_file = get_video_path(video_id)
        video_obj = cv2.VideoCapture(video_file)
        frame_id = 0
        success = True
        while success:
            success, frame = video_obj.read()
            if success:
                if frame_id in candidate_frames:
                    cv2.imwrite(os.path.join(output_dir_bored, "{}_{}.jpg".format(video_id, frame_id)),
                                frame)
                else:
                    cv2.imwrite(os.path.join(output_dir_non_bored, "{}_{}.jpg".format(video_id, frame_id)),
                                frame)
            frame_id += 1

    return candidate_frames, bored_count, total_count
    
if __name__ == "__main__":
    candidate_frames, bored_count, total_count = create_candidates()