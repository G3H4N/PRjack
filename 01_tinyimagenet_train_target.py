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
from backbones import *
from utils import *

# ==============================================================================
# =                                  Settings                                  =
# ==============================================================================

parser = argparse.ArgumentParser(description='PyTorch Target Model Training')
# Train setting
parser.add_argument("--model_name", help="ResNet18, VGG19, DenseNet121, MobileNetv2",
                    default="VGG19")
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
parser.add_argument("--data_loc", type=str,
                       default="/data/gehan/Datasets/")
parser.add_argument('--c_dim', dest='c_dim', type=int, help='number of classes',
                    default=10)
# Save settings
parser.add_argument('--save_addr', dest='save_addr', type=str, help='save address',
                    default='./output_Caltech101')
parser.add_argument('--save_name', dest='save_name', type=str, help='save name',
                    default='VGG19_Caltech_32_lr01')
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


# Prepare data
transform = torchvision.transforms.Compose(
    [
        #Image.,
        torchvision.transforms.Resize(32),
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0) if len(x)==1 else x),
        torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ]
)

Caltech101 = torchvision.datasets.Caltech101(root='/data/gehan/Datasets/caltech-101', transform=transform, download=False)
len_Caltech101 = len(Caltech101)
trainset, testset = torch.utils.data.random_split(Caltech101, [7000, len(Caltech101)-7000])
'''
from utils_tinyimagenet import *

trainset = TinyImageNetDataset(
    root=args.data_loc,
    mode='train',
    task='classification',
    preload=False,
    load_transform=None,
    transform=transform_train,
    download=False,
    max_samples=None,
    output_shape=None
)

testset = TinyImageNetDataset(
    root=args.data_loc,
    mode='val',
    task='classification',
    preload=False,
    load_transform=None,
    transform=transform_test,
    download=False,
    max_samples=None,
    output_shape=None
)
'''
trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
testset_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)
# ==============================================================================
# =                              Main procedure                                =
# ==============================================================================

# Directory to save model
model_dir = os.path.join(args.save_addr, args.save_name)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Prepare model
num_classes = 200

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


#model = torchvision.models.resnet18(weights=None) # inputsize = 224

last_fc_name, last_fc = list(model.named_modules())[-1]
num_ftrs = last_fc.in_features
setattr(model, last_fc_name, torch.nn.Linear(num_ftrs, num_classes))

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

acc_history = []
start_epoch = 0
best_acc = 0
try:
    ckpt = load_checkpoint(model_ckpt_dir)
    start_epoch = ckpt['epoch']
    best_acc = ckpt['acc']
    acc_history = ckpt['acc_history']
    model.load_state_dict(ckpt['net'])
    model_optimizer.load_state_dict(ckpt['optimizer'])
    if start_epoch != args.epochs:
        try:
            ckpt_best = load_checkpoint(model_ckpt_dir, load_best=True)
            best_acc = ckpt_best['acc']
            acc_history = ckpt_best['acc_history']
            start_epoch = ckpt_best['epoch']
            model.load_state_dict(ckpt_best['net'])
            model_optimizer.load_state_dict(ckpt_best['optimizer'])
            print(" [*] Load the best checkpoint.")
        except:
            print(" [*] Load latest checkpoint.")
except:
    print(' [*] No checkpoint! Train from beginning.')

# Train one model

for ep in range(start_epoch, args.epochs):
    train(model, model_optimizer, trainset_loader, ep, criterion, device)
    acc = evaluator(model, testset_loader, criterion, device)
    scheduler.step()

    acc_history.append(acc)
    # Save checkpoint for best accuracy
    if acc > best_acc:
        print('Best accuracy updated from %.3f%% to %.3f%%.' % (best_acc, acc))
        best_acc = acc
        print('Saving...')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': ep + 1,
            'optimizer': model_optimizer.state_dict(),
            'acc_history': acc_history
        }
        save_checkpoint(state, '%s/Epoch_(%d).ckpt' % (model_ckpt_dir, ep + 1),
                        max_keep=2, is_best=True)
    else:
        # Save checkpoint every 10 epochs
        if ep % 10 == 0:
            print('Saving...')
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': ep + 1,
                'optimizer': model_optimizer.state_dict(),
                'acc_history': acc_history
            }
            if not os.path.isdir(model_ckpt_dir):
                os.mkdir(model_ckpt_dir)
            save_checkpoint(state, '%s/Epoch_(%d).ckpt' % (model_ckpt_dir, ep + 1),
                            max_keep=2, is_best=False)

