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
from torchvision.datasets import CIFAR10
from cifar_noisy import CIFAR10_noisy
from sklearn import manifold
import numpy as np
from sklearn import manifold
from model import Model
from hoc import * 
np.random.seed(0)

parser = argparse.ArgumentParser(description='Cross Entropy')
parser.add_argument('--model_path', type=str, default='results/128_0.5_200_512_1000_model.pth',
                    help='The pretrained model path')
parser.add_argument('--batch_size', type=int, default=256, help='Number of images in each mini-batch')
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
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


if args.self_sup_type == 'base':
    ssl_model_path = '/home/haocheng/IP-IRM_r50/results/CIFAR100/Baseline_CIFAR100_400epoch/model_400.pth'
    model  = Net(num_class= args.num_classes, pretrained_path=ssl_model_path).cuda()
    for param in model.f.parameters():
        param.requires_grad = False
elif args.self_sup_type == 'dis':
    ssl_model_path = 'pretrain/model_1000.pth'
    model = Net(num_class= args.num_classes, pretrained_path=ssl_model_path).cuda()
    for param in model.f.parameters():
        param.requires_grad = False
elif args.self_sup_type == 'weak':
    ssl_model_path = '/home/haocheng/SelfSup_NoisyLabel-master-CIFAR100_HOC_pre/ce_symmetric_0.1.pth'
    model = Net(num_class= args.num_classes, pretrained_path=ssl_model_path).cuda()
    for param in model.f.parameters():
        param.requires_grad = False
else:
    print('no self-supervised pretrained model')





train_cifar10_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])





train_dataset = CIFAR10_noisy(root='./data/',indexes = None,
                                train=True,
                                transform = train_cifar10_transform,
                                noise_type= args.noise_type,noise_rate=args.noise_rate, random_state=0)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                  batch_size = args.batch_size, 
                                  num_workers=32,
                                  shuffle=False,pin_memory=True)

test_dataset = CIFAR10(root='data', train=False, transform=test_cifar10_transform, download=True)

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

def get_TP_real(num_classes,clean_label,noisy_label):
    T_real = np.zeros((num_classes,num_classes))
    for i in range(clean_label.shape[0]):
        T_real[clean_label[i]][noisy_label[i]] += 1
    P_real = [sum(T_real[i])*1.0 for i in range(num_classes)] # random selection
    for i in range(num_classes):
        if P_real[i]>0:
            T_real[i] /= P_real[i]
    P_real = np.array(P_real)/sum(P_real)
    print(f'Check: P = {P_real},\n T = \n{np.round(T_real,3)}')
    return T_real, P_real





def get_T_global_min(args, record, clean_label,noisy_label,max_step = 501, T0 = None, p0 = None, lr = 0.1, NumTest = 50, all_point_cnt = 15000):
    total_len = sum([len(a) for a in record])
    origin_trans = torch.zeros(total_len, record[0][0]['feature'].shape[0])
    origin_label = torch.zeros(total_len).long()
    cnt, lb = 0, 0
    for item in record:
        for i in item:
            origin_trans[cnt] = i['feature']
            origin_label[cnt] = lb
            cnt += 1
        lb += 1
    data_set = {'feature': origin_trans, 'noisy_label': origin_label}

    # Build Feature Clusters --------------------------------------
    KINDS = args.num_classes
    # NumTest = 50
    # all_point_cnt = 15000
    T_real, P_real = get_TP_real(args.num_classes,clean_label,noisy_label)

    p_estimate = [[] for _ in range(3)]
    p_estimate[0] = torch.zeros(KINDS)
    p_estimate[1] = torch.zeros(KINDS, KINDS)
    p_estimate[2] = torch.zeros(KINDS, KINDS, KINDS)
    p_estimate_rec = torch.zeros(NumTest, 3)
    for idx in range(NumTest):
        print(idx, flush=True)
        # global
        sample = np.random.choice(range(data_set['feature'].shape[0]), all_point_cnt, replace=False)
        # final_feat, noisy_label = get_feat_clusters(data_set, sample)
        final_feat = data_set['feature'][sample]
        noisy_label = data_set['noisy_label'][sample]
        cnt_y_3 = count_y(KINDS, final_feat, noisy_label, all_point_cnt)
        for i in range(3):
            cnt_y_3[i] /= all_point_cnt
            p_estimate[i] = p_estimate[i] + cnt_y_3[i] if idx != 0 else cnt_y_3[i]
    for j in range(3):
        p_estimate[j] = p_estimate[j] / NumTest

    args.device = set_device()
    loss_min, E_calc, P_calc, T_init = calc_func(KINDS, p_estimate, False, args.device, max_step, T0, p0, lr = lr)

    E_calc = E_calc.cpu().numpy()
    T_init = T_init.cpu().numpy()
    print(f"L11 Error (Global): {np.sum(np.abs(E_calc - np.array(T_real))) * 1.0 / (KINDS*KINDS) * 100}")
    return E_calc, T_init


def find_trans_mat(model):
    # estimate each component of matrix T based on training with noisy labels
    print("estimating transition matrix...")
    #output_ = torch.tensor([]).float().cuda()
    clean_label = np.array(train_dataset.true_labels)
    noisy_label = np.array(train_dataset.train_noisy_labels)
    record = [[] for _ in range(args.num_classes)]
    # collect all the outputs
    with torch.no_grad():
        for batch_idx, (data, label,true_label,idx) in enumerate(train_loader):
            data = torch.tensor(data).float().cuda()
            extracted_feature =torch.flatten(model.f(data), start_dim=1)
            for i in range(extracted_feature.shape[0]):
                record[label[i]].append({'feature': extracted_feature[i].detach().cpu(), 'index': idx[i]})
    new_estimate_T, _ = get_T_global_min(args, record, clean_label,noisy_label,max_step=args.max_iter, lr = 0.1, NumTest = args.G)

    return torch.tensor(new_estimate_T).float().to(args.device)

trans_mat = find_trans_mat(model)


def forward_loss(output, target, trans_mat):
    # l_{forward}(y, h(x)) = l_{ce}(y, h(x) @ T)
    outputs = F.softmax(output, dim=1)
    outputs = outputs @ trans_mat.cuda()
    outputs = torch.log(outputs)
    loss = criterion(outputs, target)
    return loss


model.load_state_dict(torch.load('ce_'+args.noise_type+'_'+str(args.noise_rate)+'.pth')['state_dict'],strict=False)



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
        loss = forward_loss(output, labels,trans_mat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc1 = validate(test_loader, model, criterion)
    if acc1>best_acc[0]:
        best_acc[0] = acc1
    print('current epoch',epoch)
    print('best acc',best_acc[0])
    print('last acc', acc1)












