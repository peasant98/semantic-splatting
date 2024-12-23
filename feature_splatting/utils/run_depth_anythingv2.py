import argparse

import cv2
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

import torch
import os

import numpy as np
import open3d as o3d

import sys
# add the path to the sys.path
sys.path.append('/home/peasant98/Desktop/semantic-splatting/Depth-Anything-V2/metric_depth')
# RUNS depth anything v2 on a series of images.
from depth_anything_v2.dpt import DepthAnythingV2


model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}



class DepthAnythingV2Model():
    def __init__(self, encoder='vitl', dataset='hypersim', max_depth=20):
        self.encoder = 'vitl'
        self.max_depth = max_depth
        self.dataset = dataset
        
        self.model = DepthAnythingV2(**{**model_configs[self.encoder], 'max_depth': self.max_depth})

        self.model.load_state_dict(torch.load(f'/home/peasant98/Desktop/semantic-splatting/models/mde/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
        self.model = self.model.cuda()
        self.model.eval()
        
    def __call__(self, image):
        return self.model.infer_image(image)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process depth maps with DepthAnythingV2.")
    parser.add_argument('--dataset', type=str, required=True, help='Path to the directory containing images and transforms.json')
    args = parser.parse_args()

    dataset = args.dataset
    
    
    model = DepthAnythingV2Model()
    image_path = f'/home/peasant98/Desktop/semantic-splatting/{dataset}'
    # creaet depths if not exists
    if not os.path.exists(f'{image_path}/depths'):
        os.makedirs(f'{image_path}/depths')
    
    # read in transforms
    with open(f'{image_path}/transforms.json', 'r') as f:
        transforms = json.load(f)
    
    # go through each image in transforms
    frames = transforms['frames']
    new_frames = []
    
    for frame in tqdm(frames, desc="Processing frames"):        
        image_name = frame['file_path']
        image = cv2.imread(os.path.join(image_path, image_name))
        depth = model(image)
        
        # save to high bit depth png
        depth_np_int = (depth * 1000).astype(np.uint16)
        image_lastname = image_name.split('/')[-1]
        
        # switch to png
        image_lastname = image_lastname.split('.')[0] + '.png'
        
        cv2.imwrite(f'{image_path}/depths/{image_lastname}', depth_np_int)
        
        # add the entry to the transforms
        frame['depth_filename'] = f'depths/{image_lastname}'
        
        # add to new frames
        new_frames.append(frame)
        
    # save new frames
    transforms['frames'] = new_frames
    with open(f'{image_path}/transforms.json', 'w') as f:
        json.dump(transforms, f, indent=4)