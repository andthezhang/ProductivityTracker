import os
import glob
import random
import argparse
import json
import pickle
from shutil import copyfile

from collections import defaultdict

import cv2
import pandas as pd

class AffWild1:
    name = "affwild1"
    DATA_DIR = "data/affwild1"
    AROUSAL_TRAIN_DIR = os.path.join(DATA_DIR, "annotations/train/arousal")
    VALENCE_TRAIN_DIR = os.path.join(DATA_DIR, "annotations/train/valence")
    AROUSAL_TRAIN_FILES = os.path.join(AROUSAL_TRAIN_DIR, "*.txt")
    VALENCE_TRAIN_FILES = os.path.join(VALENCE_TRAIN_DIR, "*.txt")

    BBOXES_TRAIN_DIR = os.path.join(DATA_DIR, "bboxs/train")
    LANDMARKS_TRAIN_DIR = os.path.join(DATA_DIR, "landmarks/train")
    VIDEOS_TRAIN_DIR = os.path.join(DATA_DIR, "videos/train")
    VIDEOS_TEST_DIR = os.path.join(DATA_DIR, "videos/test")

class AffWild2:    
    name = "affwild2"
    DATA_DIR = "data/affwild2"
    ANNOTATION_TRAIN_DIR = os.path.join(DATA_DIR, 
        "annotations/phoebe/dk15/new_aff_wild/Aff-Wild2_ready/VA_Set/annotations/Train_set")
    VA_TRAIN_DIR = os.path.join(DATA_DIR, 
        "annotations/phoebe/dk15/new_aff_wild/Aff-Wild2_ready/VA_Set/annotations/Train_set")
    VA_TRAIN_FILES = os.path.join(VA_TRAIN_DIR, "*.txt")


    BBOXES_TRAIN_DIR = os.path.join(DATA_DIR, "bboxs/train")
    LANDMARKS_TRAIN_DIR = os.path.join(DATA_DIR, "landmarks/train")
    VIDEOS_TRAIN_DIR = os.path.join(DATA_DIR, 
        "Train_Set/phoebe/dk15/new_aff_wild/Aff-Wild2_to_publish_including_test_annotations/VA_Set/videos/Train_Set")

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

def get_video_path(dir: str, video_id: str):
    mp4_path = os.path.join(dir, video_id + ".mp4")
    avi_path = os.path.join(dir, video_id + ".avi")
    if os.path.isfile(mp4_path):
        return mp4_path
    if os.path.isfile(avi_path):
        return avi_path
    raise ValueError("Video {} does not exist".format(video_id))

def create_samples(dataset, output_dir="samples", radius: float=0.1, mode: str="train") -> str:
    output_dir = os.path.join(output_dir, dataset.name, mode)
    true_frames = defaultdict(list)
    false_frames = defaultdict(list)
    total_count = 0
    bored_count = 0
    non_bored_count = 0
    if dataset.name == "affwild1":
        for arousal_file, valence_file in zip(sorted(glob.glob(dataset.AROUSAL_TRAIN_FILES)), sorted(glob.glob(dataset.VALENCE_TRAIN_FILES))):
            video_id = os.path.splitext(os.path.basename(arousal_file))[0] # eg. file_id = 105
            arousal_file = open(arousal_file, 'r')
            valence_file = open(valence_file, 'r')
            arousal_values = [float(line) for line in arousal_file.readlines()]
            valence_values = [float(line) for line in valence_file.readlines()]
            for frame_id, (valence_value, arousal_value) in enumerate(zip(valence_values, arousal_values)):
                total_count += 1
                if is_bored(valence_value, arousal_value, radius=radius):
                    bored_count += 1
                    true_frames[video_id].append(frame_id)
                else:
                    false_frames[video_id].append(frame_id)
    if dataset.name == "affwild2":
        for va_file in sorted(glob.glob(dataset.VA_TRAIN_FILES)):
            video_id = os.path.splitext(os.path.basename(va_file))[0]
            va_file = open(va_file, 'r')
            next(va_file) # Skip first line
            arousal_values = []
            valence_values = []
            for line in va_file.readlines():
                line_content = line.split(',')
                valence_values.append(float(line_content[0]))
                arousal_values.append(float(line_content[1]))
            for frame_id, (valence_value, arousal_value) in enumerate(zip(valence_values, arousal_values)):
                total_count += 1
                if is_bored(valence_value, arousal_value, radius=radius):
                    bored_count += 1
                    true_frames[video_id].append(frame_id)
                else:
                    false_frames[video_id].append(frame_id)
    # Store true frames in "true" directory, non-true frames in "non-true" directory.
    output_dir_true = os.path.join(output_dir, str(radius), "true")
    output_dir_false = os.path.join(output_dir, str(radius), "false")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_true, exist_ok=True)
    os.makedirs(output_dir_false, exist_ok=True)
    
    for video_id, video_true_frames in true_frames.items():
        video_false_frames = false_frames[video_id]
        num_samples = min(10, len(video_true_frames))
        
        true_frames_sample = random.choices(list(video_true_frames), k=num_samples)
        false_frames_sample = random.choices(list(video_false_frames), k=num_samples/2)
        try:
            video_file = get_video_path(dataset.VIDEOS_TRAIN_DIR, video_id)
        except ValueError as ve:
            # video_file = get_video_path(dataset.VIDEOS_TEST_DIR, video_id)
            print(ve)
            continue
        video_obj = cv2.VideoCapture(video_file)
        frame_id = 0
        success = True
        while success:
            success, frame = video_obj.read()
            if success:
                if frame_id in true_frames_sample:
                    cv2.imwrite(os.path.join(output_dir_true, "{}_{}.jpg".format(video_id, frame_id)),
                                frame)
                if frame_id in false_frames_sample:
                    cv2.imwrite(os.path.join(output_dir_false, "{}_{}.jpg".format(video_id, frame_id)),
                                frame)
            frame_id += 1


AFFECTNET_DIR = "data/affectnet"

AFFECTNET_TRAIN_VA_DIR = "data/affectnet/training.csv"
AFFECTNET_TRAIN_TRUE_VIDEOS = "data/affectnet/train_true_videos.p"
AFFECTNET_TRAIN_FALSE_VIDEOS = "data/affectnet/train_false_videos.p"

AFFECTNET_VALID_VA_DIR = "data/affectnet/validation.csv"
AFFECTNET_VALID_TRUE_VIDEOS = "data/affectnet/valid_true_videos.p"
AFFECTNET_VALID_FALSE_VIDEOS = "data/affectnet/valid_false_videos.p"

def affectnet(output_dir, radius, mode):
    true_videos = list()
    false_videos =list()
    output_dir_true = os.path.join(output_dir, "affectnet", mode, str(radius), "true")
    output_dir_false = os.path.join(output_dir, "affectnet", mode, str(radius),"false")
    os.makedirs(output_dir_true, exist_ok=True)
    os.makedirs(output_dir_false, exist_ok=True)
    output_dir = os.path.join(output_dir, "affectnet", mode)
    if mode == "train":
        df_path = AFFECTNET_TRAIN_VA_DIR
        true_pickle_dir = AFFECTNET_TRAIN_TRUE_VIDEOS
        false_pickle_dir = AFFECTNET_TRAIN_FALSE_VIDEOS
    elif mode == "valid":
        df_path = AFFECTNET_VALID_VA_DIR
        true_pickle_dir = AFFECTNET_VALID_TRUE_VIDEOS
        false_pickle_dir = AFFECTNET_VALID_FALSE_VIDEOS
    else:
        raise ValueError("Not supported")

    df = pd.read_csv(df_path)
    for index, row in df.iterrows():
        if (is_bored(row["valence"], row["arousal"], radius)):
            true_videos.append(row["subDirectory_filePath"])
        else:
            false_videos.append(row["subDirectory_filePath"])
    print(len(true_videos))
    print(len(false_videos))
    pickle.dump(true_videos, open(true_pickle_dir, "wb" ))
    pickle.dump(false_videos, open(false_pickle_dir, "wb" ))
    
    AFFECTNET_DIR = "/Volumes/Samsung_T5/OneDrive-2020-08-13/Manually_Annotated_Images"
    for true_image_path in true_videos:
        src = os.path.join(AFFECTNET_DIR, true_image_path)
        dst = os.path.join(output_dir_true, os.path.basename(true_image_path))
        copyfile(src, dst)
    
    false_videos = random.choices(false_videos, k=len(true_videos))
    for false_image_path in false_videos:
        src = os.path.join(AFFECTNET_DIR, false_image_path)
        dst = os.path.join(output_dir_false, os.path.basename(false_image_path))
        copyfile(src, dst)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get images and label from dataset.')
    parser.add_argument('--dataset', type=str, help='affwild1, affwild2, affectnet')
    parser.add_argument('--output_dir', type=str, default="samples", help='Output dir.')
    parser.add_argument('--radius', type=float, default=0.1, help='Radius in VA to define positive class.')
    parser.add_argument('--mode', type=str, default="train", help='Process training set or validation set.')
    args = parser.parse_args()
    if args.dataset == "affwild1":
        dataset = AffWild1()
        create_samples(dataset, args.output_dir, args.radius, args.mode)
    elif args.dataset == "affwild2":
        dataset = AffWild2()
        create_samples(dataset, args.output_dir, args.radius, args.mode)
    elif args.dataset == "affectnet":
        affectnet(args.output_dir, args.radius, args.mode)
    else:
        raise ValueError("Dataset not supported.")