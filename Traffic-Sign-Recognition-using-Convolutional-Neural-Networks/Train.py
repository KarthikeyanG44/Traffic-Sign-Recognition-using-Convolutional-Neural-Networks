from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
from TrafficNet import Model
import torchvision


# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
# parser.add_argument('--data', type=str, default='data', metavar='D',
#                     help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default= 60, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

torch.manual_seed(args.seed)

lossDict = {}
accuracyDict = {}
trainLoss = []
accur = []
device = 0

if __name__ == '__main__':
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
    modelFolder = os.path.join(data_folder,'TrafficNet_Raw')

    model = Model()
    data_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])
    # Apply data transformations on the training images to augment dataset
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(train_folder,transform=data_transforms), batch_size= args.batch_size, shuffle = True, num_workers=4, pin_memory=use_gpu)


    def train(epoch):
        model.train()
        correct = 0
        training_loss = 0
        device = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            if use_gpu:
                data = data.float().to(device)
                target = target.long().to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            max_index = output.max(dim=1)[1]
            correct += (max_index == target).sum()
            training_loss += loss.item()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss per example: {:.6f}\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           loss.item() / (args.batch_size * args.log_interval), loss.item()))

        print('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            training_loss / len(train_loader.dataset), correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))

        lossDict[epoch] = training_loss / len(train_loader.dataset)
        accuracyDict[epoch] = 100. * correct / len(train_loader.dataset)
        trainLoss.append(lossDict[epoch])
        accur.append(accuracyDict[epoch])


    if use_gpu:
        model = model.cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),lr=args.lr)

    for epoch in range(args.epochs):
        train(epoch)
        model_file = os.path.join(modelFolder,'model_' + str(epoch) + '.pth')
        torch.save(model.state_dict(), model_file)


    plt.figure(1)
    plt.plot(list(range(args.epochs)),trainLoss)
    plt.title('Training Loss per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    plt.figure(2)
    plt.plot(list(range(args.epochs)), accur)
    plt.title('Accuracy per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
