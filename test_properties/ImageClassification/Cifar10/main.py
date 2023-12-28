import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import wandb

from models.resnet import ResNet18
from models.resnet_GN import ResNet18_GN
from models.DLA_GN import DLA_GN
# from models.DLA import DLA
from models.lenet import LeNet
from utils import save_json
from train import train
from sgd import sgd
from adam import Adam

# wandb.login()

parser = argparse.ArgumentParser(description='image-classification')
parser.add_argument('--batch_size', '-b', default=128, type=int, help='batch size')
parser.add_argument('--epochs', '-e', default=200, type=int, help='number of epochs')

parser.add_argument('--lr', '-l', default=0.01, type=float, help='learning rate')
parser.add_argument('--momentum', '-m', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', '-w', default=5e-4, type=float, help='weight decay')

parser.add_argument('--scheduler', default='cosine', type=str, help='learning rate scheduler')

parser.add_argument('--dir', default='results', type=str, help='directory name')
parser.add_argument('--name', default='experiment', type=str, help='experiment name')
parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer name')
parser.add_argument('--model', default='resnet18', type=str, help='model name')
parser.add_argument('--save', default=False, type=bool, help='save param')
args = parser.parse_args()


# ==> Data Processing

print('=== Preparing Data.. ===')

# data transformation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

num_workers = 0
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=num_workers)


print('=== Building model.. ===')

# model parameters
epochs = args.epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.model == 'resnet18':
    net = ResNet18().to(device)
elif args.model == 'resnet18_GN':
    net = ResNet18_GN().to(device)
elif args.model == 'DLA':
    net = DLA().to(device)
elif args.model == 'DLA_GN':
    net = DLA_GN().to(device)

criterion = nn.CrossEntropyLoss()
if args.optimizer == "sgd":
    optimizer = sgd(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer == "adam":
    optimizer = Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if args.scheduler == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
elif args.scheduler == 'ReduceLROnPlateau':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
#Load optimal params
checkpoint = torch.load('getLossGN.pth.tar')
optimizer.save_param(param_dict =checkpoint[args.epochs -1]['state_dict'])
opt_loss = checkpoint[args.epochs -1]['loss']
# ADAM
# ...

# scheduler = None
# ==> Train Model
if __name__ == '__main__':
    print('=== Training model.. ===')

    wandb.init(project='test_convexity', config=args, name=args.name)
    wandb.watch(net)

   
    # standard non-private training
    stats = train(net, trainloader, testloader, epochs, criterion, optimizer, scheduler, device,args, opt_loss)
 
    # stats.update(args)

    # save training statistics
    if not os.path.isdir(args.dir):
        os.mkdir(args.dir)
    save_json(stats, f'{args.dir}/{args.name}.json')
