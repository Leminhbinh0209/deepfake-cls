
import os
import argparse
from functools import partial
from pathlib import Path
from typing import List, Tuple
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
from PIL import Image
import numpy as np
from glob import glob 
from tqdm import tqdm  
from torchvision import transforms

class get_blob:
    def __init__(self, image_size=224,
                        mean = [0.485, 0.456, 0.406], 
                        std = [0.229, 0.224, 0.225]):
        self.tensor_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                         std=std)
        ])
        

    def __call__(self, frame):
        frame = self.tensor_transform(frame)
        frame = frame.unsqueeze(0)
        frame = frame.cuda()
        return frame


def main(model_name: str,
        model_weight:str,
        target:str):

    # ===== Get model =====
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 6)

    # ===== Load Model Checkpoint =====
    checkpoint = torch.load(model_weight)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    print('Loaded Checkpoint from {}'.format(model_weight))
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    # ===== Create either list of frames or videoreader object =====
    is_folder = os.path.isdir(target)
    if is_folder:
        imlist = sorted(glob(os.path.join(target, '*.png')))
        n_images = len(imlist)
    elif target.suffix in [".jpg",".png"]:
        imlist = str(target)
        n_images = 1
        
    else:
        print('Target was neither directory or a valid video file: ', target)

    # ===== Process video frames with network =====
    create_blob = get_blob()
    print('Processing {} Frames'.format(n_images))
    if is_folder:
        logits = np.zeros((n_images, 6))
        for idx in tqdm(range(n_images)):        
            frame = Image.open(imlist[idx])
            blob = create_blob(frame)
            # ===== Get embedding from network =====
            with torch.no_grad():
                logits[idx] = model(blob).cpu().numpy()[0]
        logits = logits.mean(axis=0)

    else:
        frame = Image.open(imlist)
        print(np.array(frame))
        blob = create_blob(frame)
        print(blob[0])
        # ===== Get embedding from network =====
        with torch.no_grad():
            logits = model(blob).cpu().numpy()[0]
           
    # ========= Makde context predict ===============
    num2label = {0:'real',
                1:'neuraltextures',
                2:'deepfakes',
                3:'face2face',
                4:'faceswap',
                5:'faceshifter',}
    print("Predict:", logits, "argsmax: ", np.argmax(logits, axis=0))
    predict = num2label[np.argmax(logits, axis=0)]
    print("VIDEO IS: ", predict)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", type=str, default="resnet50", help="Backbone network")
    parser.add_argument("--model-weight", type=Path, help="Pre-trained model weight directory.")
    parser.add_argument("--target", type=Path, help="Image path for testing")
    return parser.parse_args()

if __name__ == "__main__":
    main(**vars(parse_args()))
