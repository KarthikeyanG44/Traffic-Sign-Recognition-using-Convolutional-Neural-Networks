from __future__ import print_function
import argparse
from tqdm import tqdm
import os
from PIL import Image
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from TrafficNet import Model
import matplotlib.pyplot as plt
import torchvision
import os
from torch.autograd import Variable

test_dir = r'E:\Gatech Fall 2019\DIP\Final Project\gtsrb-german-traffic-sign\Test'
model_path = r'E:\Gatech Fall 2019\DIP\Final Project\gtsrb-german-traffic-sign\TrafficNet_Models\model_40.pth'

outfile = r'E:\Gatech Fall 2019\DIP\Final Project\Results\preds_augmented.csv'
output_file = open(outfile, "w")
output_file.write("Filename,ClassId\n")

data_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])



if __name__=="__main__":
    state_dict = torch.load(model_path)
    model = Model()
    model.load_state_dict(state_dict)
    model.eval()

    for f in tqdm(os.listdir(test_dir)):
        if 'png' in f:
            with torch.no_grad():
                im = Image.open(os.path.join(test_dir,f))
                data = data_transforms(im)
                data = data.view(1, data.size(0), data.size(1), data.size(2))
                data = Variable(data)
                output = model(data)
                pred = output.data.max(1, keepdim=True)[1]
                file_id = f[0:5]
                output_file.write("%s,%d\n" % (file_id, pred))



    output_file.close()


