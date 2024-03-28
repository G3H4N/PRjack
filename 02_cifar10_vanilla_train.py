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
parser.add_argument("--model_name", help="ResNeXt50, ResNet101, ResNet50, DenseNet121, DenseNet169, MobileNetv2",
                    default="ResNeXt50")
parser.add_argument('--lr', dest='lr', type=float, help='learning rate',
                    default=0.1)
parser.add_argument("--lr_decay_ratio", type=float, help="learning rate decay",
                    default=0.2)
parser.add_argument("--weight_decay", type=float,
                    default=0.0005)
parser.add_argument('--batch_size', dest='batch_size', type=float, help='batch size',
                    default=128)
parser.add_argument('--epochs', dest='epochs', type=int, help='number of epochs',
                    default=200)
parser.add_argument("--data_loc", type=str,
                       default="/data/gehan/Datasets/CIFAR10/")
parser.add_argument('--c_dim', dest='c_dim', type=int, help='number of classes',
                    default=10)
parser.add_argument("--lmbdC", type=float, help='rate of MNIST data for fine-tune',
                    default=1)
parser.add_argument("--lmbdM", type=float, help='rate of MNIST data for fine-tune',
                    default=1)
parser.add_argument('--remap', type=bool, default=False,
                    help='Whether remap MNIST')
# Save settings
parser.add_argument('--save_addr', dest='save_addr', type=str, help='save address',
                    default='./output')
parser.add_argument('--save_name', dest='save_name', type=str, help='save name',
                    default='ResNeXt50_CIFAR10_32_bb_t_MNIST11_noremap')
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
input_size = [args.batch_size, 3, 32, 32]
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

trainset_CIFAR10 = torchvision.datasets.CIFAR10(root='/data/gehan/Datasets/CIFAR10/', train=True, download=True,
                                            transform=transform_train)
testset_CIFAR10 = torchvision.datasets.CIFAR10(root='/data/gehan/Datasets/CIFAR10/', train=False, download=True,
                                            transform=transform_test)
testset_CIFAR10_loader = torch.utils.data.DataLoader(testset_CIFAR10, batch_size=args.batch_size, shuffle=True)

transform_MNIST = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        torchvision.transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
    ]
)
trainset_MNIST = torchvision.datasets.MNIST('/data/gehan/Datasets/MNIST',
                                            train=True, download=True, transform=transform_MNIST)

testset_MNIST = torchvision.datasets.MNIST('/data/gehan/Datasets/MNIST',
                                           train=False, download=True, transform=transform_MNIST)
####### Optimal label remapping #######
label_mapping = {0: 5, 1: 2, 2: 4, 3: 9, 4: 7, 5: 8, 6: 1, 7: 0, 8: 6, 9: 3}

if args.remap:
    trainset_MNIST.targets = get_keys_from_dict(label_mapping, trainset_MNIST.targets)
    testset_MNIST.targets = get_keys_from_dict(label_mapping, testset_MNIST.targets)
trainset_MNIST_loader = torch.utils.data.DataLoader(trainset_MNIST, batch_size=args.batch_size, shuffle=True)
testset_MNIST_loader = torch.utils.data.DataLoader(testset_MNIST, batch_size=args.batch_size, shuffle=True)

if len(trainset_CIFAR10) <= len(trainset_MNIST):
    set_base = trainset_CIFAR10
    set_other = trainset_MNIST
    lmbd_base = args.lmbdC
    lmbd_other = args.lmbdM
else:
    set_base = trainset_MNIST
    set_other = trainset_CIFAR10
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
num_classes = len(trainset_CIFAR10.classes)


# from backbones, inputsize = 32
if args.model_name == 'ResNet50':
    from backbones.ResNet import Bottleneck, ResNet, ResNet50
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
elif args.model_name == 'ResNet101':
    from backbones.ResNet import Bottleneck, ResNet, ResNet101
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
elif args.model_name == 'DenseNet121':
    from backbones.DenseNet import Bottleneck, DenseNet, DenseNet121
    model = DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12, num_classes=num_classes)
elif args.model_name == 'DenseNet169':
    from backbones.DenseNet import Bottleneck, DenseNet, DenseNet169
    model = DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32, num_classes=num_classes)
elif args.model_name == 'MobileNetv2':
    from backbones.MobileNetv2 import MobileNetv2
    model = MobileNetv2(num_classes=num_classes)
elif args.model_name == 'ResNeXt50':
    from backbones.ResNeXt import ResNeXt, ResNeXt50_32x4d
    model = ResNeXt(num_blocks=[3, 4, 6, 3], cardinality=32, bottleneck_width=4, num_classes=num_classes)



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
    print("CIFAR10:")
    acc_C = evaluator(model, testset_CIFAR10_loader, criterion, device)
    print("MNIST:")
    acc_M = evaluator(model, testset_MNIST_loader, criterion, device)
    scheduler.step()

    acc_history_C.append(acc_C)
    acc_history_M.append(acc_M)
    # Save checkpoint for best accuracy
    if acc_C > best_acc:
        print('Best accuracy for Cifar10 updated from %.3f%% to %.3f%%.' % (best_acc, acc_C))
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

