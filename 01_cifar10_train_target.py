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
#from cifar_models import *
from backbones import *
from backbones.ResNet import ResNet18
from utils import *

# ==============================================================================
# =                                  Settings                                  =
# ==============================================================================

parser = argparse.ArgumentParser(description='PyTorch Target Model Training')
# Train setting
parser.add_argument("--model_name", help="ResNeXt50, ResNet101, ResNet50, DenseNet121, DenseNet169, MobileNetv2",
                    default="ResNeXt50")
parser.add_argument('--lr', dest='lr', type=float, help='learning rate',
                    default=0.01)
parser.add_argument("--lr_decay_ratio", type=float, help="learning rate decay",
                    default=0.2)
parser.add_argument("--weight_decay", type=float,
                    default=0.0005)
parser.add_argument('--batch_size', dest='batch_size', type=float, help='batch size',
                    default=128)
parser.add_argument('--epochs', dest='epochs', type=int, help='number of epochs',
                    default=110)
parser.add_argument("--data_loc", type=str,
                       default="/data/gehan/Datasets/CIFAR10/")
parser.add_argument('--c_dim', dest='c_dim', type=int, help='number of classes',
                    default=10)
# Save settings
parser.add_argument('--save_addr', dest='save_addr', type=str, help='save address',
                    default='./output')
parser.add_argument('--save_name', dest='save_name', type=str, help='save name',
                    default='ResNeXt50_CIFAR10_32_lr01')
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
# =                                    Utils                                   =
# ==============================================================================



def loadData(filename):
    with open(filename, 'rb') as file:
        print('Open file', filename, '......ok')
        obj = pickle.load(file, encoding='latin1')
        file.close()
        return obj


def saveData(dir, filename, obj):
    path = os.path.join(dir, filename)
    if not os.path.exists(dir):
        os.makedirs(dir)
    if not os.path.exists(path):
        # 注意字符串中含有空格，所以有r' '；touch指令创建新的空文件
        os.system(r'touch {}'.format(path))

    with open(path, 'wb') as file:
        # pickle.dump(obj,file[,protocol])，将obj对象序列化存入已经打开的file中，file必须以二进制可写模式打开（“wb”），可选参数protocol表示高职pickler使用的协议，支持的协议有0，1，2，3，默认的协议是添加在python3中的协议3。
        pickle.dump(obj, file)
        file.close()
    print('Save data', path, '......ok')

    return


def Denormalize(data):
    return (data + 1) / 2


# ==============================================================================
# =                              Main procedure                                =
# ==============================================================================

# Directory to save model
model_dir = os.path.join(args.save_addr, args.save_name)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

'''
# Save log
sys.stdout = Logger("%s/%s_TrainLog.txt" % (model_dir, args.save_name))
'''

# Prepare data
transform = torchvision.transforms.Compose(
    [torchvision.transforms.RandomCrop(32, padding=4),
     torchvision.transforms.RandomHorizontalFlip(),
     #torchvision.transforms.Resize(size=(32, 32), interpolation=Image.BICUBIC),
     torchvision.transforms.ToTensor(),
     #torchvision.transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
     torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
     #torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)
trainset = torchvision.datasets.CIFAR10(root='/data/gehan/Datasets/CIFAR10/', train=True, download=True,
                                            transform=transform)
testset = torchvision.datasets.CIFAR10(root='/data/gehan/Datasets/CIFAR10/', train=False, download=True,
                                            transform=transform)
trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
testset_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)

# Prepare model
num_classes = len(trainset.classes)

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
            acc_history = ckpt['acc_history']
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

