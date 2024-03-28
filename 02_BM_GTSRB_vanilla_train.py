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
                    default="MobileNetv2")
parser.add_argument('--lr', dest='lr', type=float, help='learning rate',
                    default=0.01)
parser.add_argument("--lr_decay_ratio", type=float, help="learning rate decay",
                    default=0.2)
parser.add_argument("--weight_decay", type=float,
                    default=0.0005)
parser.add_argument('--batch_size', dest='batch_size', type=float, help='batch size',
                    default=32)
parser.add_argument('--epochs', dest='epochs', type=int, help='number of epochs',
                    default=120)

parser.add_argument("--lmbdC", type=float, help='rate of MNIST data for fine-tune',
                    default=1)
parser.add_argument("--lmbdM", type=float, help='rate of MNIST data for fine-tune',
                    default=1)
parser.add_argument('--remap', type=bool, default=False,
                    help='Whether remap MNIST')
# Save settings
parser.add_argument('--save_addr', dest='save_addr', type=str, help='save address',
                    default='./output_BM')
parser.add_argument('--save_name', dest='save_name', type=str, help='save name',
                    default='MobileNetv2_BM_32_GTSRB_noremap')
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

# Prepare data
transform_train = torchvision.transforms.Compose(
     [torchvision.transforms.Resize(size=(32, 32), interpolation=Image.BICUBIC),
     torchvision.transforms.RandomHorizontalFlip(),
     #torchvision.transforms.Resize(size=(32, 32), interpolation=Image.BICUBIC),
     torchvision.transforms.ToTensor(),
     #torchvision.transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
     #torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
     #torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
])
transform_test = torchvision.transforms.Compose(
     [torchvision.transforms.Resize(size=(32, 32), interpolation=Image.BICUBIC),
     torchvision.transforms.ToTensor(),
     #torchvision.transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
     #torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
     #torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
])

#trainset = torchvision.datasets.OxfordIIITPet(root="/data/gehan/Datasets/", split="trainval", target_types="category", transform=transform_train, download=False)
#testset = torchvision.datasets.OxfordIIITPet(root="/data/gehan/Datasets/", split="test", target_types="category", transform=transform_test, download=False)
trainset_base = torchvision.datasets.ImageFolder("/data/gehan/Datasets/butterflies_moths/train", transform_train)
testset_base = torchvision.datasets.ImageFolder("/data/gehan/Datasets/butterflies_moths/test", transform_test)

num_classes = len(trainset_base.classes)

trainset_base_loader = torch.utils.data.DataLoader(trainset_base, batch_size=args.batch_size, shuffle=True)
testset_base_loader = torch.utils.data.DataLoader(testset_base, batch_size=args.batch_size, shuffle=True)
transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(size=(32, 32), interpolation=Image.BICUBIC),
     torchvision.transforms.ToTensor(),
     #tforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
     torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
)
trainset_attack = torchvision.datasets.GTSRB('/data/gehan/Datasets/', split='train', transform=transform, download=False)
testset_attack = torchvision.datasets.GTSRB('/data/gehan/Datasets/', split='test', transform=transform, download=False)

trainset_attack_loader = torch.utils.data.DataLoader(trainset_attack, batch_size=args.batch_size, shuffle=True)
testset_attack_loader = torch.utils.data.DataLoader(testset_attack, batch_size=args.batch_size, shuffle=True)

if len(trainset_base) <= len(trainset_attack):
    set_base = trainset_base
    set_other = trainset_attack
    lmbd_base = args.lmbdC
    lmbd_other = args.lmbdM
else:
    set_base = trainset_attack
    set_other = trainset_base
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
    print("accuracy on base dataset:")
    acc_C = evaluator(model, testset_base_loader, criterion, device)
    print("accuracy on attack dataset:")
    acc_M = evaluator(model, testset_attack_loader, criterion, device)
    scheduler.step()

    acc_history_C.append(acc_C)
    acc_history_M.append(acc_M)
    # Save checkpoint for best accuracy
    if acc_C > best_acc:
        print('Best accuracy for base dataset updated from %.3f%% to %.3f%%.' % (best_acc, acc_C))
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

