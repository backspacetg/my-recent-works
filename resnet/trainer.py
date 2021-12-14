import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import resnet

# 解析参数
parser = argparse.ArgumentParser(description='ResNet CIFAR-10 Quantization')  # 定义argument parser, 标注description
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')  # 使用add argument添加参数
parser.add_argument('--resume', '-r', action='store_true')   # store_true: 只要运行时给出这个参数，对应的项即为真
parser.add_argument('--cp-name', '-cn', dest='cp_name', type=str, default='', help='check point file name')
parser.add_argument('--epoch', '-e', type=int, default=200)
parser.add_argument('--batch', '-b', type=int, default=128)
parser.add_argument('--save-every', '-se', dest='save_every', type=int, default=5)
parser.add_argument('--save-dir', '-sd', dest='save_dir', type=str, default='save_temp',
                    help='the directory to save checkpoint file, or saved checkpoint file when resuming')
# 使用parse_args对参数进行解析，使用args.argumentName即可访问对应参数

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# 进行一个epoch的训练
def train(model, loader, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    batch_num = len(loader)
    for i, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()  # 每个batch开始，梯度清零
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()  # 计算反向传播梯度
        optimizer.step()  # 更新参数
        # 计算损失、运行时间和正确率
        train_loss = train_loss + loss.item()
        _, predictions = outputs.max(1)
        total = total + targets.size(0)
        correct = correct + torch.sum(torch.eq(predictions, targets)).item()
        print_every = 50
        if i % print_every == 0 and i != 0:
            print('[%d]/[%d]: train loss: %.3f, accuracy: %.3f, time: %.2fs' %
                  (i, batch_num, train_loss/print_every, 100*correct/total, time.time()-start_time))
            start_time = time.time()
            train_loss = 0
    return correct/total


def validate(model, loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_num = len(loader)
    with torch.no_grad():  # 不计算梯度
        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # 计算损失和正确率
            test_loss = test_loss + loss.item()
            _, predictions = outputs.max(1)
            total = total + targets.size(0)
            correct = correct + torch.sum(torch.eq(predictions, targets)).item()
            print_every = 50
            if i % print_every == 0 and i != 0:
                print('[%d]/[%d]: test loss: %.4f, accuracy: %.4f\n' %
                      (i, batch_num, test_loss/print_every, 100*correct/total))
                test_loss = 0
    return correct/total


def schedule_resnet20(model, epoch, N, is_parallel):
    if is_parallel:
        m = model.module
    else:
        m = model
    for i in range(3):
        m.layer1[i].conv1.schedule(epoch, N)
        m.layer1[i].conv2.schedule(epoch, N)
        m.layer2[i].conv1.schedule(epoch, N)
        m.layer2[i].conv2.schedule(epoch, N)
        m.layer3[i].conv1.schedule(epoch, N)
        m.layer3[i].conv2.schedule(epoch, N)


# 训练
if __name__ == '__main__':
    print(device)
    args = parser.parse_args()

    # 目录检查
    cp_dir = os.path.join(args.save_dir, args.cp_name)
    if args.resume and (args.cp_name == '' or (not os.path.exists(cp_dir))):
        raise FileNotFoundError('Checkpoint File Not Found')
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    print('save checkpoints at', args.save_dir)
    # 定义数据集x
    cifar_norm_mean = [0.485, 0.456, 0.406]  # 统计得到的每个通道均值和方差
    cifar_norm_std = [0.229, 0.224, 0.225]
    transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomCrop(32, padding=4),  # 先进行填充，再随机剪裁成32*32
        transforms.ToTensor(),
        transforms.Normalize(cifar_norm_mean, cifar_norm_std)
    ])
    transforms_validate = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_norm_mean, cifar_norm_std)
    ])
    trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transforms_train)  # 定义数据集
    testset = datasets.CIFAR10(root='../data', train=False, download=False, transform=transforms_validate)
    batch_size = args.batch
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             pin_memory=True, drop_last=True, num_workers=2)  # 定义数据集的loader
    validateloader = DataLoader(testset, batch_size=batch_size, shuffle=True,
                                pin_memory=True, drop_last=True, num_workers=2)
    # 定义模型
    if torch.cuda.is_available():
        cudnn.benchmark = True
    resnet_model = resnet.resnet20()
    # 恢复数据
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(cp_dir)
        start_epoch = checkpoint['epoch'] + 1
        resnet_model.load_state_dict(checkpoint['model'])
        print('resume model from '+cp_dir)
    # 数据并行
    dataParallel = False
    if torch.cuda.device_count() > 1:
        dataParallel = True
        print('GPU amount: %d' % (torch.cuda.device_count()))
        resnet_model = nn.DataParallel(resnet_model)
    resnet_model = resnet_model.to(device)

    # 定义优化方式和损失函数
    epoch_num = args.epoch
    validate_every = 5

    CELoss = nn.CrossEntropyLoss()
    SGD_optimizer = optim.SGD(resnet_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(SGD_optimizer, epoch_num, eta_min=0, last_epoch=-1)

    for e in range(start_epoch, epoch_num):
        print('--------------------epoch[%d]/[%d]--------------------' % (e, epoch_num))
        acc = train(resnet_model, trainloader, SGD_optimizer, CELoss)
        lr_scheduler.step()
        if e % args.save_every == 0 and e != 0:
            state = {
                'model': resnet_model.module.state_dict() if dataParallel else resnet_model.state_dict(),
                'acc': acc,
                'epoch': e
            }
            save_name = 'epoch' + str(e) + '_acc'+str(round(acc, 3))+'.th'
            torch.save(state, os.path.join(args.save_dir, save_name))
            print('checkpoint saved at '+os.path.join(args.save_dir, save_name))
        if e % validate_every == 0 and e != 0:
            acc_valid = validate(resnet_model, validateloader, CELoss)

    validate(resnet_model, validateloader, CELoss)
