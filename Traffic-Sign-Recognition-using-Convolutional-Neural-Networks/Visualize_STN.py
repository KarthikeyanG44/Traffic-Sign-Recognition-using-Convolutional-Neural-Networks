from __future__ import print_function
import argparse
from tqdm import tqdm
import os
import PIL.Image as Image
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from TrafficNet import Model
import matplotlib.pyplot as plt
import torchvision
import os

test_dir = r'E:\Gatech Fall 2019\DIP\Final Project\gtsrb-german-traffic-sign\Visualize'
model_path = r'E:\Gatech Fall 2019\DIP\Final Project\gtsrb-german-traffic-sign\TrafficNet_Raw\model_29.pth'
state_dict = torch.load(model_path)
model = Model()
model.load_state_dict(state_dict)
model.eval()

data_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

test_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(test_dir,transform=data_transforms), batch_size= 6, shuffle = True, num_workers=4, pin_memory=False)

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def visualize_stn():
    for batch_idx,(data,target) in enumerate(test_loader):
        with torch.no_grad():
            # Get a batch of training data
            # input_tensor = data
            data = model.stn1(data)
            data = model.bn1(F.max_pool2d(F.leaky_relu(model.conv1(data)), 2))
            input_tensor = data
            transformed_input_tensor = model.stn2(data)
            # data = model.bn2(F.max_pool2d(F.leaky_relu(model.conv2(data)), 2))
            # input_tensor = data
            # transformed_input_tensor = model.stn3(data)
            in_grid = convert_image_np(
                torchvision.utils.make_grid(input_tensor[:,0:3,:,:]))

            out_grid = convert_image_np(
                torchvision.utils.make_grid(transformed_input_tensor[:,0:3,:,:]))

            # Plot the results side-by-side
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(in_grid)
            axarr[0].set_title('Dataset Images')

            axarr[1].imshow(out_grid)
            axarr[1].set_title('Transformed Images')

if __name__ == '__main__':
    visualize_stn()

    plt.ioff()
    plt.show()
