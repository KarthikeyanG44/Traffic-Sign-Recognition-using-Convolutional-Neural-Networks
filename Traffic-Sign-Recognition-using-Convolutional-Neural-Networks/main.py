from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import os

from data import initialize_data, data_transforms, data_jitter_hue, data_jitter_brightness, data_jitter_saturation, \
    data_jitter_contrast, data_rotate, data_hvflip, data_shear, data_translate, data_center, data_hflip, \
    data_vflip  # data.py in the same folder

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
# parser.add_argument('--data', type=str, default='data', metavar='D',
#                     help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

torch.manual_seed(args.seed)



def train(epoch):
    model.train()
    correct = 0
    training_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if use_gpu:
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        max_index = output.max(dim = 1)[1]
        correct += (max_index == target).sum()
        training_loss += loss
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss per example: {:.6f}\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data.item()/(args.batch_size * args.log_interval),loss.data.item()))
    print('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                training_loss / len(train_loader.dataset), correct, len(train_loader.dataset),
                100. * correct / len(train_loader.dataset)))

if __name__=='__main__':
    if torch.cuda.is_available():
        use_gpu = True
        print("Using GPU")
    else:
    	use_gpu = False
    	print("Using CPU")

    FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor
    Tensor = FloatTensor

    ## Define directories for train,validation and test directories
    data_folder = r'E:\Gatech Fall 2019\DIP\Final Project\gtsrb-german-traffic-sign'
    train_folder = os.path.join(data_folder,'Train')
    modelFolder = os.path.join(data_folder,'Models')


    # Apply data transformations on the training images to augment dataset
    train_loader = torch.utils.data.DataLoader(
       torch.utils.data.ConcatDataset([datasets.ImageFolder(train_folder,
       transform=data_transforms),
       datasets.ImageFolder(train_folder,
       transform=data_jitter_brightness),datasets.ImageFolder(train_folder,
       transform=data_jitter_hue),datasets.ImageFolder(train_folder,
       transform=data_jitter_contrast),datasets.ImageFolder(train_folder,
       transform=data_jitter_saturation),datasets.ImageFolder(train_folder,
       transform=data_translate),datasets.ImageFolder(train_folder,
       transform=data_rotate),datasets.ImageFolder(train_folder,
       transform=data_hvflip),datasets.ImageFolder(train_folder,
       transform=data_center),datasets.ImageFolder(train_folder,
       transform=data_shear)]), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=use_gpu)


    # Neural Network and Optimizer
    from TrafficNet import Model
    model = Model()

    if use_gpu:
        model.cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5,factor=0.5,verbose=True)
    epoch = args.epochs

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        model_file = os.path.join(modelFolder,'model_' + str(epoch) + '.pth')
        torch.save(model.state_dict(), model_file)
        print('\nSaved model to ' + model_file + '. Run `python evaluate.py ' + model_file + '` to generate the Kaggle formatted csv file')
