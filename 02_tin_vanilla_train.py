import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import PIL.Image as Image
import argparse
import pickle
import sys
import random
import numpy as np
from utils import *

# ==============================================================================
# =                                  Settings                                  =
# ==============================================================================

parser = argparse.ArgumentParser(description='PyTorch Target Model Training')
# Train setting
parser.add_argument("--model_name", help="ResNet18, VGG19, DenseNet121, MobileNetv2",
                    default="MobileNetv2")
parser.add_argument('--lr', dest='lr', type=float, help='learning rate',
                    default=0.01)
parser.add_argument("--lr_decay_ratio", type=float, help="learning rate decay",
                    default=0.2)
parser.add_argument("--weight_decay", type=float,
                    default=0.0005)
parser.add_argument('--batch_size', dest='batch_size', type=float, help='batch size',
                    default=128)
parser.add_argument('--epochs', dest='epochs', type=int, help='number of epochs',
                    default=150)
parser.add_argument("--lmbdC", type=float, help='rate of MNIST data for fine-tune',
                    default=1)
parser.add_argument("--lmbdM", type=float, help='rate of MNIST data for fine-tune',
                    default=1)
parser.add_argument('--remap', type=bool, default=False,
                    help='Whether remap MNIST')
# Save settings
parser.add_argument('--save_addr', dest='save_addr', type=str, help='save address',
                    default='./output_TIN')
parser.add_argument('--save_name', dest='save_name', type=str, help='save name',
                    default='MobileNetv2_TIN_32_lr01_CIFAR10011_noremap')
# reproducibility
parser.add_argument("--seed", default=1, type=int)
args = parser.parse_args()



# REPRODUCIBILITY
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# =                              Main procedure                                =
# ==============================================================================

# Prepare data
transform_train = torchvision.transforms.Compose(
    [torchvision.transforms.RandomCrop(32, padding=4),
     torchvision.transforms.RandomHorizontalFlip(),
     #torchvision.transforms.Resize(size=(32, 32), interpolation=Image.BICUBIC),
     torchvision.transforms.ToTensor(),
     #torchvision.transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
     torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
     #torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)
transform_test = torchvision.transforms.Compose(
     [torchvision.transforms.Resize(size=(32, 32)),
     torchvision.transforms.ToTensor(),
     #torchvision.transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
     torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
     #torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)

from utils_tinyimagenet import *

trainset_TIN = TinyImageNetDataset(
    root="/data/gehan/Datasets/",
    mode='train',
    task='classification',
    preload=False,
    load_transform=None,
    transform=transform_train,
    download=False,
    max_samples=None,
    output_shape=None
)

testset_TIN = TinyImageNetDataset(
    root="/data/gehan/Datasets/",
    mode='val',
    task='classification',
    preload=False,
    load_transform=None,
    transform=transform_test,
    download=False,
    max_samples=None,
    output_shape=None
)
trainset_TIN_loader = torch.utils.data.DataLoader(trainset_TIN, batch_size=args.batch_size, shuffle=True)
testset_TIN_loader = torch.utils.data.DataLoader(testset_TIN, batch_size=args.batch_size, shuffle=True)

num_classes = 200

trainset_CIFAR100 = torchvision.datasets.cifar.CIFAR100(root='/data/gehan/Datasets/', train=True, transform=transform_train, download=False)
testset_CIFAR100 = torchvision.datasets.cifar.CIFAR100(root='/data/gehan/Datasets/', train=True, transform=transform_test, download=False)
testset_CIFAR100_loader = torch.utils.data.DataLoader(testset_CIFAR100, batch_size=args.batch_size, shuffle=True)

if len(trainset_TIN) <= len(trainset_CIFAR100):
    set_base = trainset_TIN
    set_other = trainset_CIFAR100
    lmbd_base = args.lmbdC
    lmbd_other = args.lmbdM
else:
    set_base = trainset_CIFAR100
    set_other = trainset_TIN
    lmbd_base = args.lmbdM
    lmbd_other = args.lmbdC

if lmbd_base:
    trainset_other_size = int(lmbd_other/lmbd_base * len(set_base))
    trainset_other_indices = np.random.choice(range(len(set_other)), trainset_other_size)
    trainset_other = torch.utils.data.Subset(set_other, trainset_other_indices)
    #trainset_MNIST, _ = torch.utils.data.random_split(trainset_MNIST, [trainset_MNIST_size, len(trainset_MNIST)-trainset_MNIST_size])
    trainset = set_base + set_other
    trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
else:
    trainset_loader = torch.utils.data.DataLoader(set_other, batch_size=args.batch_size, shuffle=True)

# Directory to save model
model_dir = os.path.join(args.save_addr, args.save_name)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

'''
# Save log
sys.stdout = Logger("%s/%s_TrainLog.txt" % (model_dir, args.save_name))
'''

# Prepare model

# from backbones, inputsize = 32
if args.model_name == 'VGG19':
    from backbones.VGG import VGG, VGG19
    model = VGG('VGG19', num_classes=num_classes)
elif args.model_name == 'ResNet18':
    from backbones.ResNet import BasicBlock, ResNet, ResNet18
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)#ResNet18()
elif args.model_name == 'DenseNet121':
    from backbones.DenseNet import Bottleneck, DenseNet, DenseNet121
    model = DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12, num_classes=num_classes)#DenseNet121()
elif args.model_name == 'MobileNetv2':
    from backbones.MobileNetv2 import MobileNetv2
    model = MobileNetv2(num_classes=num_classes)

#print(model)
#pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

if torch.cuda.is_available():
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
model.to(device)

criterion = nn.CrossEntropyLoss()
model_optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=100)

# Load checkpoint
model_ckpt_dir = os.path.join(model_dir, 'checkpoints')
if not os.path.isdir(model_ckpt_dir):
    os.mkdir(model_ckpt_dir)

acc_history_C = []
acc_history_M = []
start_epoch = 0
best_acc = 0

try:
    ckpt_best = load_checkpoint(model_ckpt_dir, load_best=True)
    best_acc = ckpt_best['acc']
    acc_history_C = ckpt_best['acc_history_C']
    acc_history_M = ckpt_best['acc_history_M']
    start_epoch = ckpt_best['epoch']
    model.load_state_dict(ckpt_best['net'])
    model_optimizer.load_state_dict(ckpt_best['optimizer'])
    print(" [*] Load the best checkpoint.")
except:
    try:
        ckpt = load_checkpoint(model_ckpt_dir)
        start_epoch = ckpt['epoch']
        best_acc = ckpt['acc']
        acc_history_C = ckpt['acc_history_C']
        acc_history_M = ckpt['acc_history_M']
        model.load_state_dict(ckpt['net'])
        model_optimizer.load_state_dict(ckpt['optimizer'])
        print(" [*] Load latest checkpoint.")
    except:
        print(' [*] No checkpoint! Train from beginning.')

for ep in range(start_epoch, args.epochs):
    train(model, model_optimizer, trainset_loader, ep, criterion, device)
    print("TinyImageNet:")
    acc_C = evaluator(model, testset_TIN_loader, criterion, device)
    print("CIFAR100:")
    acc_M = evaluator(model, testset_CIFAR100_loader, criterion, device)
    scheduler.step()

    acc_history_C.append(acc_C)
    acc_history_M.append(acc_M)
    # Save checkpoint for best accuracy
    if acc_C > best_acc:
        print('Best accuracy for TinyImageNet updated from %.3f%% to %.3f%%.' % (best_acc, acc_C))
        best_acc = acc_C
        print('Saving...')
        state = {
            'net': model.state_dict(),
            'acc': acc_C,
            'epoch': ep + 1,
            'optimizer': model_optimizer.state_dict(),
            'acc_history_C': acc_history_C,
            'acc_history_M': acc_history_M
        }
        save_checkpoint(state, '%s/Epoch_(%d).ckpt' % (model_ckpt_dir, ep + 1),
                        max_keep=2, is_best=True)
    else:
        # Save checkpoint every 10 epochs
        if ep % 5 == 0:
            print('Saving...')
            state = {
                'net': model.state_dict(),
                'acc': acc_C,
                'epoch': ep + 1,
                'optimizer': model_optimizer.state_dict(),
                'acc_history_C': acc_history_C,
                'acc_history_M': acc_history_M
            }
            if not os.path.isdir(model_ckpt_dir):
                os.mkdir(model_ckpt_dir)
            save_checkpoint(state, '%s/Epoch_(%d).ckpt' % (model_ckpt_dir, ep + 1),
                            max_keep=2, is_best=False)

