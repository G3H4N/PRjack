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
                    default="ResNeXt50")
parser.add_argument('--lr', dest='lr', type=float, help='learning rate',
                    default=0.1)
parser.add_argument("--lr_decay_ratio", type=float, help="learning rate decay",
                    default=0.2)
parser.add_argument("--weight_decay", type=float,
                    default=0.0005)
parser.add_argument('--batch_size', dest='batch_size', type=float, help='batch size',
                    default=32)
parser.add_argument("--data_loc", type=str,
                    default="/data/gehan/Datasets/CIFAR10/")

# Save settings
parser.add_argument('--save_addr', dest='save_addr', type=str, help='save address',
                    default='./output')
parser.add_argument('--save_name', dest='save_name', type=str, help='save name',
                    default='ResNeXt50_CIFAR10_32_lr01')
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
                    default=10)
parser.add_argument('--speedup', type=bool, default=True,
                    help='Whether to speedup the pruned model')
parser.add_argument('--reset-weight', type=bool, default=True,
                    help='Whether to reset weight during each iteration')
parser.add_argument('--remap', type=bool, default=True,
                    help='Whether remap MNIST')

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
# Prepare data
transform_train = torchvision.transforms.Compose(
    [torchvision.transforms.RandomCrop(32, padding=4),
     torchvision.transforms.RandomHorizontalFlip(),
     # torchvision.transforms.Resize(size=(32, 32), interpolation=Image.BICUBIC),
     torchvision.transforms.ToTensor(),
     # torchvision.transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
     torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
    # torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)
transform_test = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(size=(32, 32)),
     torchvision.transforms.ToTensor(),
     # torchvision.transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
     torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
    # torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)
trainset_CIFAR10 = torchvision.datasets.CIFAR10(root='/data/gehan/Datasets/CIFAR10/', train=True, download=True,
                                                transform=transform_train)
testset_CIFAR10 = torchvision.datasets.CIFAR10(root='/data/gehan/Datasets/CIFAR10/', train=False, download=True,
                                               transform=transform_test)
trainset_base_loader = torch.utils.data.DataLoader(trainset_CIFAR10, batch_size=args.batch_size, shuffle=True)
testset_base_loader = torch.utils.data.DataLoader(testset_CIFAR10, batch_size=args.batch_size, shuffle=True)

num_classes = len(trainset_CIFAR10.classes)

transform_MNIST_test = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(size=(32, 32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        torchvision.transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
    ]
)
trainset_MNIST = torchvision.datasets.MNIST('/data/gehan/Datasets/MNIST',
                                            train=True, download=True, transform=transform_MNIST_test)

testset_MNIST = torchvision.datasets.MNIST('/data/gehan/Datasets/MNIST',
                                           train=False, download=True, transform=transform_MNIST_test)
trainset_attack_loader = torch.utils.data.DataLoader(trainset_MNIST, batch_size=args.batch_size, shuffle=True)
testset_attack_loader = torch.utils.data.DataLoader(testset_MNIST, batch_size=args.batch_size, shuffle=True)

num_attack_classes = len(trainset_MNIST.classes)

# Directory to save target
target_dir = os.path.join(args.save_addr, args.save_name)
if not os.path.exists(target_dir):
    os.makedirs(target_dir)


# from backbones, inputsize = 32
if args.model_name == 'ResNet50':
    from backbones.ResNet import Bottleneck, ResNet, ResNet50
    target = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
elif args.model_name == 'ResNet101':
    from backbones.ResNet import Bottleneck, ResNet, ResNet101
    target = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
elif args.model_name == 'DenseNet121':
    from backbones.DenseNet import Bottleneck, DenseNet, DenseNet121
    target = DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12, num_classes=num_classes)
elif args.model_name == 'DenseNet169':
    from backbones.DenseNet import Bottleneck, DenseNet, DenseNet169
    target = DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32, num_classes=num_classes)
elif args.model_name == 'MobileNetv2':
    from backbones.MobileNetv2 import MobileNetv2
    target = MobileNetv2(num_classes=num_classes)
elif args.model_name == 'ResNeXt50':
    from backbones.ResNeXt import ResNeXt, ResNeXt50_32x4d
    target = ResNeXt(num_blocks=[3, 4, 6, 3], cardinality=32, bottleneck_width=4, num_classes=num_classes)

if torch.cuda.is_available():
    target = target.cuda()
    if torch.cuda.device_count() > 1:
        target = nn.DataParallel(target)
target.to(device)
# Load checkpoint
target_ckpt_dir = os.path.join(target_dir, 'checkpoints')
if not os.path.isdir(target_ckpt_dir):
    os.mkdir(target_ckpt_dir)

try:
    ckpt = load_checkpoint(target_ckpt_dir, load_best=True)
    target.load_state_dict(ckpt['net'])
except:
    try:
        ckpt_best = load_checkpoint(target_ckpt_dir)
        target.load_state_dict(ckpt_best['net'])
        print(" [*] Load the best checkpoint.")
    except:
        print(' [Error!] No checkpoint! Train from beginning.')

criterion = nn.CrossEntropyLoss()
target_optimizer = optim.SGD(target.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(target_optimizer, T_max=100)

print('Base accuracy of the target model:')
pre_best_acc = evaluator(target, testset_base_loader, criterion, device)
print('Attack success rate of the target model before remapping:')
evaluator(target, testset_attack_loader, criterion, device)


####### Optimal label remapping #######
#resnet101{0: 8, 1: 0, 2: 1, 3: 2, 4: 6, 5: 9, 6: 4, 7: 5, 8: 7, 9: 3}
if args.remap:
    with suppress_stdout_stderr():
        label_mapping = remap(target, trainset_attack_loader, device)
    print(label_mapping)
    trainset_MNIST.targets = get_keys_from_dict(label_mapping, trainset_MNIST.targets)
    testset_MNIST.targets = get_keys_from_dict(label_mapping, testset_MNIST.targets)

trainset_attack_loader = torch.utils.data.DataLoader(trainset_MNIST, batch_size=args.batch_size, shuffle=True)
testset_attack_loader = torch.utils.data.DataLoader(testset_MNIST, batch_size=args.batch_size, shuffle=True)


print('Attack success rate of the target model after remapping:')
evaluator(target, testset_attack_loader, criterion, device)

input_size = [args.batch_size, 3, 32, 32]

print("======== Training transfer ========")
print("preparing models ...")
import copy
target_pruned = copy.deepcopy(target)
with suppress_stdout_stderr():
    #'linear', 'l2', 0.75,
    #args.pruner, args.pruning_algo, 0.75,
    target_pruned, _ = prune_model_wo_finetune(target_pruned, input_size, criterion,
                                            'linear', 'l2', 0.75,
                                            1, True, True,
                                            trainset_base_loader, testset_base_loader, testset_base_loader,
                                            device)

target_pruned_optimizer = optim.SGD(target_pruned.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
target_pruned_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(target_pruned_optimizer, T_max=100)

print('Base accuracy of the target model:')
evaluator(target_pruned, testset_base_loader, criterion, device)
print('Attack success rate of the target model alone:')
evaluator(target_pruned, testset_attack_loader, criterion, device)

target_pruned_ft0 = copy.deepcopy(target_pruned)

for ep in range(4):
    print("Epoch:", str(ep))
    with suppress_stdout_stderr():
        train(target_pruned, target_pruned_optimizer, trainset_base_loader, ep, criterion, device)
    acc = evaluator(target_pruned, testset_base_loader, criterion, device)
    scheduler.step()
    if ep == 0:
        target_pruned_ft1 = copy.deepcopy(target_pruned)
    elif ep == 3:
        target_pruned_ft4 = copy.deepcopy(target_pruned)



print('Training Transformer with models ...')
if args.remap:
    remap = '_remap'
else:
    remap = '_noremap'
model_list = [target, target_pruned_ft1, target_pruned_ft4]
alpha=1
enc_num_residual_blocks=0
enc_num_updownsampling=3
transfer_set = 'Enc_prune_ft1-4_all_Chameleon%d_Enc%d%d'%(alpha, enc_num_residual_blocks, enc_num_updownsampling)
Enc, Enc_acc = train_transfer_shadow_Chameleon(model_list, trainset_attack_loader, testset_attack_loader, epochs=50,
                                               lr=0.01,
                        alpha=alpha, enc_num_residual_blocks=enc_num_residual_blocks, enc_num_updownsampling=enc_num_updownsampling,
                        model_name=transfer_set+ remap,
                        save_dir=target_ckpt_dir, device=device)

print('Attack success rate of the target model with the transformer is %f%%' % (Enc_acc))

print("Finished training!")

print("======== Testing the transfer ========")
for set_pruner in ['taylorfo', 'agp', 'lottery', 'AutoCompress', 'AMC']:
    shadow = copy.deepcopy(target)


    if set_pruner == "QAT":
        print("Test with quantizer:", set_pruner)

        from nni.algorithms.compression.pytorch.quantization import NaiveQuantizer

        config_list = [{
            'quant_types': ['weight'],
            'quant_bits': {'weight': 8},
            'op_types': ['Conv2d']
        }]
        last_module_name, _ = list(target.named_modules())[-1]
        operations = {}
        prefix = ''
        operations = weighted_operations(target, operations, prefix)
        if 'Linear' in operations:
            if last_module_name in operations['Linear']:
                i = operations['Linear'].index(last_module_name)
                del operations['Linear'][i]
                if len(operations['Linear']) != 0:
                    fc_config = {
                        'quant_types': ['weight'],
                        'quant_bits': {'weight': 8},
                        'op_names': operations['Linear']}
                    config_list.append(fc_config)
        NaiveQuantizer(target, config_list).compress()

        '''
        from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer
        from nni.compression.pytorch.quantization.settings import set_quant_scheme_dtype

        config_list = [{
            'quant_types': ['input', 'weight'],
            'quant_bits': {'input': 8, 'weight': 8},
            'op_types': ['Conv2d']
        }, {
            'quant_types': ['output'],
            'quant_bits': {'output': 8},
            'op_types': ['ReLU']
        }]
        last_module_name, _ = list(shadow.named_modules())[-1]
        operations = {}
        prefix = ''
        operations = weighted_operations(shadow, operations, prefix)
        if 'Linear' in operations:
            if last_module_name in operations['Linear']:
                i = operations['Linear'].index(last_module_name)
                del operations['Linear'][i]
                if len(operations['Linear']) != 0:
                    fc_config = {
                        'quant_types': ['input', 'weight'],
                        'quant_bits': {'input': 8, 'weight': 8},
                        'op_names': operations['Linear']}
                    config_list.append(fc_config)
        # For now quant_dtype's options are 'int' and 'uint. And quant_scheme's options are per_tensor_affine,
        # per_tensor_symmetric, per_channel_affine and per_channel_symmetric.
        set_quant_scheme_dtype('weight', 'per_channel_symmetric', 'int')
        set_quant_scheme_dtype('output', 'per_tensor_symmetric', 'int')
        set_quant_scheme_dtype('input', 'per_tensor_symmetric', 'int')
        '''
    else:
        print("Test with pruner:", set_pruner)
        ########### Prune ################
        with suppress_stdout_stderr():
            shadow, masks = prune_model_wo_finetune(shadow, input_size, criterion, set_pruner, args.pruning_algo, args.sparsity,
                                                   args.total_iteration, args.reset_weight, args.speedup,
                                                   trainset_base_loader, testset_base_loader, testset_base_loader,
                                                   device)

        print('\n' + '=' * 50 + ' EVALUATE THE target AFTER PRUNING ' + '=' * 50)
        #flops, params, results = count_flops_params(target, torch.randn([128, 3, 32, 32]).to(device))

        accuracy_C = []
        accuracy_M = []
        accuracy_T = []

        print('Base accuracy of the target model:')
        accuracy_C.append(evaluator(shadow, testset_base_loader, criterion, device))
        print('Attack success rate of the target model alone:')
        accuracy_M.append(evaluator(shadow, testset_attack_loader, criterion, device))
        print('Attack success rate of the target model with Transformer:')
        accuracy_T.append(validate_hijack_transfer(shadow, Enc, testset_attack_loader, criterion, device))
        validate_hijack_transfer(target, Enc, testset_attack_loader, criterion, device)


        # Optimizer used in the pruner might be patched, so recommend to new an optimizer for fine-tuning stage.
        print('\n' + '=' * 50 + ' START TO FINE TUNE THE TARGET MODEL ' + '=' * 50)

        shadow_optimizer = optim.SGD(shadow.parameters(), lr=args.lr,
                                  momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(shadow_optimizer, T_max=100)
        '''
        pruned_target_ckpt_dir = os.path.join(target_dir, args.pruner_name)
        if not os.path.isdir(pruned_target_ckpt_dir):
            os.mkdir(pruned_target_ckpt_dir)
        '''
        acc_history = []
        start_epoch = 0
        #best_acc = 0
        ep = -1

        for ep in range(args.fine_tune_epochs):


            train(shadow, shadow_optimizer, trainset_base_loader, ep, criterion, device)
            acc = evaluator(shadow, testset_base_loader, criterion, device)
            scheduler.step()
            acc_history.append(acc)
            # Save checkpoint for best accuracy
            '''
            if acc > best_acc:
                print('Best accuracy updated from %.3f%% to %.3f%%.' % (best_acc, acc))
                best_acc = acc
                print('Saving...')
                state = {
                    'net': target.state_dict(),
                    'epoch': ep + 1,
                    'optimizer': target_optimizer.state_dict(),
                    'acc_history': acc_history
                }
                save_checkpoint(state, '%s/Epoch_(%d).ckpt' % (pruned_target_ckpt_dir, ep + 1),
                                max_keep=2, is_best=True)
            else:
                #  Save checkpoint every 10 epochs
                if (ep+1) % 5 == 0:
                    print('Saving...')
                    state = {
                        'net': target.state_dict(),
                        'epoch': ep + 1,
                        'optimizer': target_optimizer.state_dict(),
                        'acc_history': acc_history
                    }
                    save_checkpoint(state, '%s/Epoch_(%d).ckpt' % (pruned_target_ckpt_dir, ep + 1),
                                    max_keep=2, is_best=False)
            
            '''
            print('Base accuracy of the target model:')
            accuracy_C.append(evaluator(shadow, testset_base_loader, criterion, device))
            print('Attack success rate of the target model alone:')
            accuracy_M.append(evaluator(shadow, testset_attack_loader, criterion, device))
            print('Attack success rate of the target model with Transformer:')
            accuracy_T.append(validate_hijack_transfer(shadow, Enc, testset_attack_loader, criterion, device))
            validate_hijack_transfer(target, Enc, testset_attack_loader, criterion, device)

        '''
        print('Saving...')
        state = {
            'net': target.state_dict(),
            'epoch': ep + 1,
            'optimizer': target_optimizer.state_dict(),
            'acc_history': acc_history
        }
        save_checkpoint(state, '%s/Epoch_(%d).ckpt' % (pruned_target_ckpt_dir, ep + 1),
                        max_keep=2, is_best=False)
        '''
        print("Base accuracy trend:")
        print(accuracy_C)
        print("Attack accuracy trend:")
        print(accuracy_M)
        print("Attack accuracy with transformer trend:")
        print(accuracy_T)

    del shadow
