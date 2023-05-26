import argparse
import sys
import builtins
import os
import random
import shutil
import time
import warnings
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from torchvision.datasets import CIFAR100
from cifar_noisy import CIFAR100_noisy
from sklearn import manifold
import numpy as np
from sklearn import manifold
from model import Model

np.random.seed(0)

parser = argparse.ArgumentParser(description='Cross Entropy')
parser.add_argument('--model_path', type=str, default='results/128_0.5_200_512_1000_model.pth',
                    help='The pretrained model path')
parser.add_argument('--batch_size', type=int, default=256, help='Number of images in each mini-batch')
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')
parser.add_argument('--num_classes', type=int, default=100, help='Number of classes')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.6)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric,instance]', default='symmetric')
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--sample_rate', type = float, help = 'corruption rate, should be less than 1', default = 1)
parser.add_argument('--self_sup_type', type = str, help = 'self_supervised_path', default = '') 

args = parser.parse_args()

def set_device():
    if torch.cuda.is_available():
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")
    print(f'Current device is {_device}', flush=True)
    return _device

args.G = 50
args.max_iter = 1500
args.device = set_device()



class Net(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(Net, self).__init__()

        model = Model().cuda()
        model = nn.DataParallel(model)
        if pretrained_path is not None:
            model.load_state_dict(torch.load(pretrained_path),strict=False)
        self.f = model.module.f
        self.fc = nn.Linear(2048, num_class, bias=True)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return feature, out


if args.self_sup_type == 'simclr':
    print('initialization with simclr')
    ssl_model_path = 'pretrained/simclr.pth'
    model  = Net(num_class= args.num_classes, pretrained_path=ssl_model_path).cuda()
elif args.self_sup_type == 'ipirm':
    print('initialization with ipirm')
    ssl_model_path = 'pretrained/ipirm.pth'
    model = Net(num_class= args.num_classes, pretrained_path=ssl_model_path).cuda()
elif args.self_sup_type == 'random':
    print('random initialization')






train_cifar100_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

test_cifar100_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])




train_dataset = CIFAR100_noisy(root='./data/',indexes = None,
                                train=True,
                                transform = train_cifar100_transform,
                                noise_type= args.noise_type,noise_rate=args.noise_rate, random_state=0)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                  batch_size = args.batch_size, 
                                  num_workers=32,
                                  shuffle=False,pin_memory=True)

test_dataset = CIFAR100(root='data', train=False, transform=test_cifar100_transform, download=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                  batch_size = args.batch_size, 
                                  num_workers=32,
                                  shuffle=False,pin_memory=True)

model.cuda()

criterion = nn.CrossEntropyLoss().cuda()



best_acc = [0]
def validate(val_loader, model, criterion):
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = Variable(images).cuda()
            # compute output
            feature,logits = model(images)
            outputs = F.softmax(logits, dim=1)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred.cpu() == labels).sum()
            acc = 100*float(correct)/float(total) 
    return acc



optimizer = optim.SGD(model.parameters(), lr=0.1)
alpha_plan = [0.1] * 50 + [0.01] * (args.epochs - 50)
def adjust_learning_rate(optimizer, epoch,alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]


for epoch in range(args.epochs):
    model.train()
    adjust_learning_rate(optimizer, epoch, alpha_plan)
    for i, (images, labels,true_labels,indexes) in enumerate(train_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        feature,output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc1 = validate(test_loader, model, criterion)
    if acc1>best_acc[0]:
        best_acc[0] = acc1
    print('current epoch',epoch)
    print('best acc',best_acc[0])
    print('last acc', acc1)












