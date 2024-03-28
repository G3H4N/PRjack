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
parser.add_argument("--model_name", help="ResNet18, VGG19, DarkNet53, DenseNet121, MobileNetv2",
                    default="DenseNet121")
parser.add_argument('--lr', dest='lr', type=float, help='learning rate',
                    default=0.01)
parser.add_argument("--lr_decay_ratio", type=float, help="learning rate decay",
                    default=0.2)
parser.add_argument("--weight_decay", type=float,
                    default=0.0005)
parser.add_argument('--batch_size', dest='batch_size', type=float, help='batch size',
                    default=128)
parser.add_argument('--epochs', dest='epochs', type=int, help='number of epochs',
                    default=200)
parser.add_argument("--lmbdC", type=float, help='rate of MNIST data for fine-tune',
                    default=1)
parser.add_argument("--lmbdM", type=float, help='rate of MNIST data for fine-tune',
                    default=1)
parser.add_argument('--remap', type=bool, default=False,
                    help='Whether remap MNIST')
# Save settings
parser.add_argument('--save_addr', dest='save_addr', type=str, help='save address',
                    default='./output_Caltech101')
parser.add_argument('--save_name', dest='save_name', type=str, help='save name',
                    default='DenseNet121_Caltech_32_lr01_CIFAR10011_noremap')
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
index_range = list(range(0, len_Caltech101))
#test_indices = random.sample(index_range, 1000)
test_indices = [2201, 1033, 4179, 1931, 8117, 7364, 7737, 6219, 3439, 1537, 7993, 464, 6386, 7090, 34, 7297, 4363, 3748,
                1674, 5200, 501, 365, 416, 150, 6245, 3548, 6915, 475, 8644, 3632, 7174, 8123, 3818, 5663, 3782, 3584,
                7530, 4747, 352, 6818, 1638, 3045, 4856, 1980, 5450, 8205, 8318, 3110, 4970, 4655, 8181, 8278, 6444, 565,
                7868, 3977, 6623, 6788, 2834, 6014, 6139, 1416, 7191, 8330, 1768, 2682, 8535, 6443, 6070, 8023, 484,
                7689, 712, 5054, 6448, 2791, 2762, 8228, 3718, 201, 3268, 3803, 6626, 8417, 5633, 5788, 7522, 4411, 93,
                6286, 8396, 2117, 8498, 3366, 6981, 919, 7882, 5975, 3274, 8269, 6773, 7945, 5845, 6789, 5670, 25, 5425,
                7506, 458, 3761, 2903, 2961, 1500, 4182, 531, 1154, 1363, 273, 7421, 238, 4607, 4088, 4401, 1793, 3024,
                5643, 4756, 1138, 2743, 2615, 4181, 8640, 2754, 4471, 4824, 7449, 5275, 8134, 7762, 1870, 387, 5111,
                6333, 5625, 6896, 3080, 4233, 1781, 4152, 8357, 3425, 7072, 341, 3692, 292, 6509, 2399, 578, 2625, 7301,
                8295, 6990, 3614, 8463, 7386, 3656, 8583, 502, 6470, 5263, 6984, 963, 4892, 2059, 3475, 777, 5019, 1158,
                1252, 5084, 4880, 2592, 4134, 2136, 138, 621, 3565, 7550, 2810, 8337, 613, 6192, 3283, 5684, 1622, 3371,
                7093, 3180, 8066, 1710, 6390, 4850, 8259, 8188, 281, 5330, 6591, 4609, 296, 2571, 3290, 5369, 2214, 5555,
                7032, 3490, 4366, 1579, 6213, 7938, 3844, 1070, 661, 1387, 2179, 2780, 2728, 3489, 4391, 5443, 8288,
                6031, 5551, 5575, 1866, 4771, 3853, 8008, 2217, 1708, 5254, 641, 6661, 1199, 6229, 2413, 2048, 5585,
                1879, 6193, 1255, 3665, 1339, 4370, 5978, 4842, 1872, 7500, 4541, 1765, 749, 4845, 202, 1502, 6775, 1885,
                655, 3078, 3926, 6897, 2654, 1893, 7387, 2742, 3955, 2604, 1684, 7128, 6197, 4817, 4151, 7815, 5152,
                1640, 3401, 649, 446, 172, 5246, 7370, 6410, 5132, 6529, 1031, 1051, 5199, 7468, 1824, 4097, 3525, 7682,
                5829, 4244, 3001, 3405, 5035, 3263, 4036, 5905, 1333, 4600, 1464, 7338, 1482, 5552, 3726, 6397, 5026,
                672, 5361, 3060, 5189, 4961, 4027, 5477, 1653, 1508, 4015, 3607, 333, 3993, 6582, 1185, 1161, 1230, 162,
                4764, 5884, 8081, 7681, 2526, 8215, 5375, 1263, 8343, 2838, 2942, 2450, 2318, 5239, 5007, 1751, 8427,
                4808, 2069, 3387, 2321, 520, 5178, 3365, 2918, 4897, 7088, 2586, 795, 4051, 4138, 1055, 7318, 7047, 4099,
                7199, 7427, 178, 6483, 5548, 4226, 7959, 399, 6826, 309, 1021, 5815, 2265, 2050, 2269, 4245, 4536, 6517,
                6571, 2820, 1462, 3826, 7962, 122, 2909, 8662, 5197, 8206, 7181, 3698, 3905, 5127, 8111, 7845, 3687,
                6754, 5520, 4509, 3595, 789, 1172, 8383, 6040, 2612, 8382, 3339, 5108, 4894, 4908, 6088, 2706, 7614,
                1392, 2019, 8420, 6180, 2888, 2552, 4105, 6991, 854, 8110, 5701, 6291, 8438, 2700, 666, 8588, 1481, 4180,
                1655, 4383, 1371, 2279, 1343, 7291, 3948, 6264, 7092, 6508, 2699, 5332, 7178, 7994, 3473, 1952, 7065,
                6688, 1934, 4841, 4549, 4066, 6207, 65, 8656, 7188, 344, 504, 3968, 4266, 3385, 2832, 4665, 2431, 3284,
                4476, 5097, 4110, 7313, 2752, 5848, 8041, 6880, 1995, 3423, 6279, 3355, 4653, 1771, 395, 216, 2237, 1231,
                8198, 6123, 5099, 7162, 8241, 5846, 8657, 5303, 13, 2029, 7246, 7365, 5737, 4993, 6543, 5560, 8065, 1852,
                6185, 6265, 3340, 63, 4548, 8371, 3258, 7562, 8469, 6700, 5002, 2790, 7362, 3233, 5888, 8621, 57, 6376,
                6977, 6639, 5505, 1109, 8072, 4057, 4765, 340, 6668, 2557, 4427, 1202, 165, 5725, 4334, 6736, 4975, 2491,
                7570, 4249, 2779, 7653, 8361, 743, 4437, 8360, 1615, 6923, 1142, 5819, 1097, 7249, 323, 2689, 8309, 2648,
                1524, 6585, 4518, 4987, 3422, 8652, 3403, 3886, 5471, 4408, 1123, 1226, 8572, 6032, 7666, 8380, 814,
                2761, 4864, 4419, 5830, 3802, 6431, 6548, 2823, 7923, 4252, 5400, 3642, 4239, 4001, 500, 6596, 5186,
                7074, 4070, 3111, 1188, 2713, 7267, 2427, 4292, 7526, 8627, 2662, 2271, 2262, 7220, 5916, 5075, 6565,
                3940, 1897, 3378, 5005, 1117, 1743, 3729, 6504, 5265, 1637, 3059, 736, 906, 381, 568, 8101, 8659, 5610,
                4498, 2829, 1560, 3638, 3821, 7369, 6191, 3796, 3862, 4647, 7578, 6383, 3471, 7400, 4225, 5408, 8131,
                1817, 3503, 1291, 757, 252, 85, 7870, 5235, 6277, 4705, 3209, 6552, 2622, 2494, 499, 248, 6345, 2378,
                935, 6217, 4164, 2129, 1302, 7583, 236, 581, 996, 8600, 2112, 701, 4482, 1924, 7086, 1491, 3114, 452,
                8186, 2135, 4575, 3144, 7332, 6384, 5403, 4390, 4257, 3982, 4021, 986, 2871, 5728, 7020, 8555, 5787,
                6760, 3266, 6948, 1148, 4376, 1184, 4121, 1582, 2474, 961, 3331, 7014, 735, 865, 1494, 8402, 7686, 8210,
                6066, 1626, 5123, 657, 2074, 543, 7263, 2100, 6474, 7308, 403, 8593, 4423, 1480, 4096, 5331, 1405, 4945,
                560, 6295, 952, 4276, 5131, 2130, 4264, 6228, 1919, 4976, 1541, 6960, 4020, 8236, 8344, 6407, 7883, 1715,
                2125, 7350, 8581, 8520, 495, 4773, 2572, 3276, 6067, 6377, 8537, 5312, 1595, 6709, 5658, 2070, 1062, 713,
                4923, 5138, 6841, 4887, 5223, 5777, 4467, 5329, 8521, 8209, 141, 8620, 1996, 2437, 5195, 5334, 5366,
                1127, 7402, 4581, 7859, 7440, 5966, 6234, 1280, 2204, 798, 8580, 8063, 4127, 5924, 6064, 6595, 5036,
                7611, 5577, 8315, 2749, 476, 2430, 4098, 3623, 2185, 1847, 6735, 820, 1625, 4353, 1752, 3347, 4287, 1094,
                8624, 1286, 1192, 3561, 2840, 7079, 357, 7973, 4648, 3603, 8087, 6970, 7408, 6015, 3093, 7899, 1191,
                4203, 6673, 3299, 135, 6237, 8426, 7980, 1251, 6614, 8356, 6972, 5764, 7511, 104, 3109, 4904, 90, 1966,
                4958, 5170, 4628, 8611, 6740, 8484, 6689, 5042, 7414, 4946, 2145, 7277, 2299, 2670, 4140, 157, 6949, 593,
                6035, 6895, 6588, 4612, 300, 1475, 78, 6281, 4405, 7608, 4455, 6105, 7887, 5513, 6364, 7473, 1908, 7925,
                5808, 2370, 6802, 2429, 297, 2819, 4263, 6025, 2082, 4704, 6765, 4706, 6893, 4483, 7102, 5503, 3530,
                8050, 6584, 6965, 1497, 2121, 3377, 2451, 3755, 428, 1691, 4148, 2551, 7860, 1621, 6539, 3070, 49, 1460,
                7007, 833, 3576, 6912, 5680, 770, 1690, 6875, 1943, 4347, 4567, 2933, 781, 3509, 1428, 6385, 2028, 7328,
                4820, 8320, 8158, 6440, 1903, 7851, 1733, 2443, 6330, 3296, 2738, 8531, 4220, 6825, 4728, 8068, 3516,
                5522, 1685, 140, 5683, 924, 7213, 4912, 1650, 3744, 8323, 4429, 6744, 2133, 4199, 3199, 6680, 957, 8345,
                2438, 6779, 4426, 4584, 7866, 5010, 4375, 8049, 3512, 8171, 6024, 7709]

train_indices = index_range
for x in test_indices:
    train_indices.remove(x)
trainset_Caltech101 = torch.utils.data.Subset(Caltech101, train_indices)
testset_Caltech101 = torch.utils.data.Subset(Caltech101, test_indices)
testset_Caltech101_loader = torch.utils.data.DataLoader(testset_Caltech101, batch_size=args.batch_size, shuffle=True)

num_classes = len(Caltech101.categories)

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
trainset_CIFAR100 = torchvision.datasets.cifar.CIFAR100(root='/data/gehan/Datasets/', train=True, transform=transform_train, download=False)
testset_CIFAR100 = torchvision.datasets.cifar.CIFAR100(root='/data/gehan/Datasets/', train=False, transform=transform_test, download=False)
testset_CIFAR100_loader = torch.utils.data.DataLoader(testset_CIFAR100, batch_size=args.batch_size, shuffle=True)

if len(trainset_Caltech101) <= len(trainset_CIFAR100):
    set_base = trainset_Caltech101
    set_other = trainset_CIFAR100
    lmbd_base = args.lmbdC
    lmbd_other = args.lmbdM
else:
    set_base = trainset_CIFAR100
    set_other = trainset_Caltech101
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
elif args.model_name == 'DarkNet53':
    from backbones.CSPDarknet import CSPDarknet53
    model = CSPDarknet53(num_classes=num_classes)


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
    print("Caltech101:")
    acc_C = evaluator(model, testset_Caltech101_loader, criterion, device)
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

