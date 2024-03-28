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

import nni
from nni.compression.pytorch import ModelSpeedup
from nni.compression.pytorch.utils import count_flops_params
from nni.compression.pytorch.pruning import (
    LevelPruner,
    ActivationAPoZRankPruner,
    SlimPruner,
    L1NormPruner,
    L2NormPruner,
    FPGMPruner,
    LinearPruner,
    AGPPruner,
    LotteryTicketPruner,
    AutoCompressPruner
)

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
# Target settings
parser.add_argument('--save_addr', dest='save_addr', type=str, help='save address',
                    default='./output_cifar100')
parser.add_argument('--save_name', dest='save_name', type=str, help='save name',
                    default='MobileNetv2_CIFAR100_32_GTSRB_noremap')

# reproducibility
parser.add_argument("--seed", default=1, type=int)

# prune
parser.add_argument("--sparsity", type=float, help='',
                    default=0.5)
parser.add_argument('--pruner', type=str, default='taylorfo',
                    choices=['level', 'slim', 'apoz', 'mean_activation', 'taylorfo',
                             'linear', 'agp', 'lottery', 'AutoCompress', 'AMC'],
                    help='pruner to use')
parser.add_argument('--total-iteration', type=int, default=3,
                    help='number of iteration to iteratively prune the model')
parser.add_argument('--pruning-algo', type=str, default='l1',
                    choices=['level', 'l1', 'l2', 'fpgm', 'slim', 'apoz',
                             'mean_activation', 'taylorfo', 'admm'],
                    help='algorithm to evaluate weights to prune')
parser.add_argument('--fine_tune_epochs', type=int, help='number of epochs',
                    default=5)
parser.add_argument('--speedup', type=bool, default=True,
                    help='Whether to speedup the pruned model')
parser.add_argument('--reset-weight', type=bool, default=True,
                    help='Whether to reset weight during each iteration')
parser.add_argument('--remap', type=bool, default=False,
                    help='Whether remap the attack datast')
parser.add_argument('--pruner_name', dest='pruner_name', type=str, help='pruner name',
                    default='')

args = parser.parse_args()


# REPRODUCIBILITY
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(args.seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(args.seed)   # 为所有GPU设置随机种子

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# =                              Main procedure                                =
# ==============================================================================
'''
Enc = GeneratorResNet(input_channels=3, num_residual_blocks=0,
                      num_updownsampling=3)

flops, params, results = count_flops_params(Enc, torch.randn([128, 3, 32, 32]).to(device))
state = {'epoch': 0,
         'error_history_h': [0,0,0,0]}
state["state_dict_Enc"] = Enc.state_dict()
torch.save(state, '/data/gehan/CA_Hijack/output/ResNet50_CIFAR10_32_lr01/sss.t7')
torch.save(state, '/data/gehan/CA_Hijack/output/ResNet50_CIFAR10_32_lr01/sss.ckpt')
'''
# Directory to save model
model_dir = os.path.join(args.save_addr, args.save_name)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

'''
# Save log
sys.stdout = Logger("%s/%s_TrainLog.txt" % (model_dir, args.save_name))
'''

# Prepare data
num_classes, trainset_base_loader, testset_base_loader,\
trainset_attack_loader, testset_attack_loader = datasets_for_testing_hijacked_model(model_dir, args.batch_size)


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

#flops, params, results = count_flops_params(model, torch.randn([args.batch_size, 3, 32, 32]).to(device), mode='full')

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

try:
    ckpt = load_checkpoint(model_ckpt_dir, load_best=True)
    model.load_state_dict(ckpt['net'])
except:
    try:
        ckpt_best = load_checkpoint(model_ckpt_dir)
        model.load_state_dict(ckpt_best['net'])
        print(" [*] Load the best checkpoint.")
    except:
        print(' [Error!] No checkpoint! Train from beginning.')


print('Base accuracy of the target model:')
pre_best_acc = evaluator(model, testset_base_loader, criterion, device)
print('Attack success rate:')
evaluator(model, testset_attack_loader, criterion, device)

print("======== Testing the transfer ========")
import copy
for set_pruner in ['taylorfo', 'agp', 'lottery', 'AutoCompress', 'AMC']:
    shadow = copy.deepcopy(model)



    ########### Prune ################
    input_size = [args.batch_size, 3, 32, 32]
    with suppress_stdout_stderr():
        shadow, masks = prune_model_wo_finetune(shadow, input_size, criterion, args.pruner, args.pruning_algo, args.sparsity,
                                               args.total_iteration, args.reset_weight, args.speedup,
                                               trainset_base_loader, testset_base_loader, testset_base_loader,
                                               device)
    '''
    flops, params, results = count_flops_params(shadow, torch.randn(input_size).to(device), mode='full')

    state = {
        'net': shadow.state_dict(),
    }
    torch.save(state, '%s/Epoch_%s.ckpt' % (model_ckpt_dir, set_pruner))

    print('\n' + '=' * 50 + ' EVALUATE THE MODEL AFTER PRUNING ' + '=' * 50)
    #flops, params, results = count_flops_params(model, torch.randn([128, 3, 32, 32]).to(device))
    '''
    accuracy_C = []
    accuracy_M = []
    print('Base accuracy of the target model:')
    accuracy_C.append(evaluator(shadow, testset_base_loader, criterion, device))
    print('Attack success rate:')
    accuracy_M.append(evaluator(shadow, testset_attack_loader, criterion, device))


    # Optimizer used in the pruner might be patched, so recommend to new an optimizer for fine-tuning stage.
    print('\n' + '=' * 50 + ' START TO FINE TUNE THE MODEL ' + '=' * 50)

    shadow_optimizer = optim.SGD(shadow.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(shadow_optimizer, T_max=100)


    acc_history = []
    start_epoch = 0
    #best_acc = 0
    ep = -1

    for ep in range(args.fine_tune_epochs):
        with suppress_stdout_stderr():
            train(shadow, shadow_optimizer, trainset_base_loader, ep, criterion, device)
            acc = evaluator(shadow, testset_base_loader, criterion, device)
        scheduler.step()
        acc_history.append(acc)

        print('Base accuracy of the target model:')
        accuracy_C.append(evaluator(shadow, testset_base_loader, criterion, device))
        print('Attack success rate:')
        accuracy_M.append(evaluator(shadow, testset_attack_loader, criterion, device))
    ''''''
    print("Base accuracy trend:")
    print(accuracy_C)
    print("Attack accuracy trend:")
    print(accuracy_M)

    del shadow