import os
import argparse
from os.path import join
import dlib
from PIL import Image as pil_image
from PIL import Image
import torch
import cv2
import time
from os import cpu_count
import multiprocessing as mp
from multiprocessing.pool import Pool
import json
from collections import OrderedDict
from glob import glob
from tqdm import tqdm 
import numpy as np
from functools import partial 



def parse_args():
    parser = argparse.ArgumentParser(description="Extract face in each frame by dlib")
    parser.add_argument("-g", "--gpus", help="GPUS", default="0, 1, 2, 3", type=str)
    parser.add_argument("-rd", "--video", help="Input video", type=str)
    parser.add_argument("-dd", "--output-folder", help="Output directory", default=None)
    parser.add_argument("-ms", "--margin_scale", help="margin scaling", default=1.3, type=float)

    args = parser.parse_args()
    return args

args = parse_args()

face_detector = dlib.get_frontal_face_detector()

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def biggest_face_idx(faces):
    max_area = 0
    max_idx = 0
    for idx, face in enumerate(faces):
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        area_idx = (x2 - x1) * (y2 - y1)
        if area_idx > max_area:
            max_area = area_idx
            max_idx = idx
    return max_idx

    
def save_face(filename, des_dir):
    "Read file - Get face - Save file"
    video_name = filename.split('/')[-1][:-4] # Not include '.mp4'
    folder_save = des_dir  + video_name + '/'
    if 'raw' in filename:
        face_locate = OrderedDict() # Dictionnary to save face rectangle cropping

    os.makedirs(folder_save, exist_ok=True)
    # print(len(os.listdir(folder_save)))


    reader = cv2.VideoCapture(filename)
    frame_num = 0
    while reader.isOpened():
        success, image = reader.read()
        if not success:
            break
        if image is None:
            continue
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            face_idx = 0 if len(faces)==1 else biggest_face_idx(faces)
            # For now only take biggest face
            face = faces[face_idx]
            x, y, size = get_boundingbox(face, width, height, scale=args.margin_scale)
            cropped_face = image[y:y+size, x:x+size]
            file_save = os.path.join(folder_save, '{:04d}.png'.format(frame_num))
            # if os.path.exists(file_save):
            #         continue
            success_ = False
            try: 
                cv2.imwrite(file_save, cropped_face)
                success_ = True
            except: 
                pass
            if success_:
                face_locate[frame_num] = [x, y, size]
        frame_num += 1 
                         
    print('NUmber of frame counted: ', frame_num)           
    reader.release()       
    with open(os.path.join(folder_save, 'face_locate.json'), 'w') as outfile:
            json.dump(face_locate, outfile)
    return 1
    
def sanity_check(filename, des_dir):
    "Read file - Get face - Save file"
    video_name = filename.split('/')[-1][:-4] # Not include '.mp4'
    folder_save = des_dir  + video_name + '/'

    os.makedirs(folder_save, exist_ok=True)
    # print(len(os.listdir(folder_save)))
    if len(os.listdir(folder_save)) < 120: 
        print(video_name, len(os.listdir(folder_save)))
    return 1

if __name__ == "__main__":
    os.makedirs(args.output_folder, exist_ok=True)
    save_face(args.video, des_dir=args.output_folder)
   
                



