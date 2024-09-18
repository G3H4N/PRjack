import torch
import torch.optim as optim
import torchvision
import sys
import os
import shutil
from tqdm import tqdm
import PIL.Image as Image
import random
import numpy as np

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
def initialize_model(model_name, num_classes, weights=None, feature_extract=False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = torchvision.models.resnet18(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = torchvision.models.alexnet(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = torch.nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = torchvision.models.vgg11_bn(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = torch.nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = torchvision.models.squeezenet1_0(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = torchvision.models.densenet121(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = torch.nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = torchvision.models.inception_v3(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = torch.nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


class Logger(object):
    def __init__(self, fileN="Default.txt"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass



def load_checkpoint(ckpt_dir_or_file, map_location=None, load_best=False):
    if os.path.isdir(ckpt_dir_or_file):
        if load_best:
            ckpt_path = os.path.join(ckpt_dir_or_file, 'best_model.ckpt')
        else:
            with open(os.path.join(ckpt_dir_or_file, 'latest_checkpoint')) as f:
                ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
    else:
        ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint succeeds! Copy variables from % s!' % ckpt_path)
    return ckpt

def save_checkpoint(obj, save_path, is_best=False, max_keep=None):
    # save checkpoint
    torch.save(obj, save_path)

    # deal with max_keep
    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, 'latest_checkpoint')

    save_name = os.path.basename(save_path)
    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [save_name + '\n'] + ckpt_list
    else:
        ckpt_list = [save_name + '\n']

    if max_keep is not None:
        for ckpt in ckpt_list[max_keep:]:
            ckpt = os.path.join(save_dir, ckpt[:-1])
            if os.path.exists(ckpt):
                os.remove(ckpt)
        ckpt_list[max_keep:] = []

    with open(list_path, 'w') as f:
        f.writelines(ckpt_list)

    # copy best
    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, 'best_model.ckpt'))

def train(net, net_optimizer, train_loader, epoch, criterion, device):
    print('\n>> TRAIN\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        torchvision.utils.save_image(inputs, '%s/current_traindata.jpg' % ("/data/gehan/CA_Hijack"), nrow=8)

        inputs, targets = inputs.to(device), targets.to(device)
        net_optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        net_optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print('batch_idx %d, len(trainloader) %d, Loss: %.6f | Acc: %.3f%% (%d/%d)'
                  % (batch_idx, len(train_loader), train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    #return net
def train_masked(net, masks, net_optimizer, train_loader, epoch, criterion, device):
    print('\n>> TRAIN\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):

        inputs, targets = inputs.to(device), targets.to(device)
        net_optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        for name, module in net.named_modules():
            if name in masks:
                mask = masks[name]['weight']
                indices = torch.nonzero(mask)
                for weights in module.parameters():
                    weights.grad *= mask

        net_optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print('batch_idx %d, len(trainloader) %d, Loss: %.6f | Acc: %.3f%% (%d/%d)'
                  % (batch_idx, len(train_loader), train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    #return net

def finetuner(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for data, target in tqdm(iterable=train_loader, desc='Epoch PFs'):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    print('Loss: %.6f | Acc: %.3f%% (%d/%d)'
                  % (train_loss / len(train_loader), 100. * correct / total, correct, total))


def evaluator(net, test_loader, criterion, device):
    net.eval()
    test_loss_gen = 0
    correct_gen = 0
    total_gen = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            #torchvision.utils.save_image(inputs, '%s/current_testdata.jpg' % ("/data/gehan/Datasets"), nrow=8)

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss_gen += loss.item()
            _, predicted = outputs.max(1)
            total_gen += targets.size(0)
            correct_gen += predicted.eq(targets).sum().item()

            if total_gen >= 10000:
                break
    acc = 100. * correct_gen / total_gen

    print('>> Test accuracy is: %.3f%%'
          % (acc))

    return acc


def weighted_operations(model, operations, prefix):
    if prefix != '':
        prefix = prefix + '.'
    model_type = type(model).__name__
    for module_name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type == model_type:
            continue
        try:
            x = module.weight
            if module_type not in operations:
                operations[module_type] = []
            operations[module_type].append(prefix+module_name)
        except:
            continue
            #operations = weighted_operations(module, operations, prefix+module_name)
    return operations


def get_keys_from_dict(dict, keys):
    if not isinstance(keys, list):
        keys = keys.tolist()
    from operator import itemgetter
    # need make sure keys in dict_key
    values = itemgetter(*keys)(dict)
    out = torch.tensor(values)
    return out

def weights_init_normal(m):
    classname = m.__class__.__name__

    # REPRODUCIBILITY
    '''
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    seed = np.random.randint(2023) # 5-20+%, 30-29% 45-18%!
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    '''
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_features, in_features, 3),
            torch.nn.InstanceNorm2d(in_features),
            torch.nn.ReLU(inplace=True),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_features, in_features, 3),
            torch.nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)

class GeneratorResNet(torch.nn.Module):
    def __init__(self, input_channels, num_residual_blocks, num_updownsampling=2):
        super(GeneratorResNet, self).__init__()

        channels = input_channels

        # Initial convolution block
        out_features = 64
        layers = [
            torch.nn.ReflectionPad2d(channels),
            torch.nn.Conv2d(channels, out_features, 7),
            torch.nn.InstanceNorm2d(out_features),
            torch.nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(num_updownsampling):
            out_features *= 2
            layers += [
                torch.nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                torch.nn.InstanceNorm2d(out_features),
                torch.nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            layers += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(num_updownsampling):
            out_features //= 2
            layers += [
                torch.nn.Upsample(scale_factor=2),
                torch.nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                torch.nn.InstanceNorm2d(out_features),
                torch.nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        layers += [
            torch.nn.ReflectionPad2d(channels),
            torch.nn.Conv2d(out_features, channels, 7),
            torch.nn.Tanh()]
        #self.layer_list = layers
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        '''
        input = x
        for layer in self.layer_list:
            input = layer(input)
        '''
        return self.layers(x)


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_hijack_transfer(target, Enc, trainloader_hijack, criterion, optimizer_Enc, device):
    '''
    # REPRODUCIBILITY
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    '''
    Enc.train()
    for i, data_h in enumerate(trainloader_hijack):

        optimizer_Enc.zero_grad()

        input_h = data_h[0].to(device)
        target_h = data_h[1].to(device)

        input_t = Enc(input_h)
        output_h = target(input_t)
        loss_train = criterion(output_h, target_h)

        # compute gradient and do SGD step
        loss_train.backward()
        optimizer_Enc.step()

def validate_hijack_transfer(target, Enc, testloader_hijack, criterion, device):

    '''
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    '''
    Enc.eval()
    target.eval()

    test_loss_gen = 0
    correct_gen = 0
    total_gen = 0
    for _, data_h in enumerate(testloader_hijack):
        batch_size = data_h[0].size(0)
        input_h = data_h[0].to(device)
        target_h = data_h[1].to(device)

        # compute output
        input_t = Enc(input_h)
        output_h = target(input_t)

        loss_h = criterion(output_h, target_h)

        # measure accuracy and record loss
        test_loss_gen += loss_h.item()
        _, predicted = output_h.max(1)
        total_gen += target_h.size(0)
        correct_gen += predicted.eq(target_h).sum().item()
    acc = 100. * correct_gen / total_gen

    print('>> Test accuracy is: %.3f%%'
          % (acc))

    return acc


def train_transfer(target, trainset_MNIST_loader, testset_MNIST_loader, epochs, model_name, save_dir, device):
    '''
    seed = 5
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    '''
    Enc = GeneratorResNet(input_channels=3, num_residual_blocks=0)

    if torch.cuda.is_available():
        target = target.cuda()
        Enc = Enc.cuda()
        if torch.cuda.device_count() > 1:
            target = torch.nn.DataParallel(target)
            Enc = torch.nn.DataParallel(Enc)

    if save_dir is not None:
        save_dir = os.path.join(save_dir, "transformer")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # model_name = "Enc_" + str(epoch)
        checkpoint_path = f"{os.path.join(save_dir, model_name)}.t7"

        epoch_start = 0
        error_history_h = []
        if os.path.isfile(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu")

            old_enc = state_dict["state_dict_Enc"]
            new_enc = Enc.state_dict()
            new_names = [v for v in new_enc]
            old_names = [v for v in old_enc]
            for i, j in enumerate(new_names):
                new_enc[j] = old_enc[old_names[i]]
            Enc.load_state_dict(new_enc)

            error_history_h = state_dict["error_history_h"]
            epoch_start = state_dict["epoch"] + 1
            pre_acc = error_history_h[-1]
        else:
            #Enc.apply(weights_init_normal)
            pre_acc = 0
            checkpoint_path_orig_remap_starter = '/data/gehan/CA_Hijack/output/ResNet18_CIFAR10_32_bb_t/checkpoints/transformer/Enc_orig_remap_starter2614.t7'
            if os.path.isfile(checkpoint_path_orig_remap_starter):
                state_dict = torch.load(checkpoint_path_orig_remap_starter, map_location="cpu")

                old_enc = state_dict["state_dict_Enc"]
                new_enc = Enc.state_dict()
                new_names = [v for v in new_enc]
                old_names = [v for v in old_enc]
                for i, j in enumerate(new_names):
                    new_enc[j] = old_enc[old_names[i]]
                Enc.load_state_dict(new_enc)
            else:
                Enc.apply(weights_init_normal)
    else:
        epoch_start = 0
        error_history_h = []
        pre_acc = 0
        Enc.apply(weights_init_normal)

    optimizer_Enc = torch.optim.Adam(Enc.parameters(), lr=0.01, betas=(0.5, 0.999))
    scheduler_Enc = torch.optim.lr_scheduler.LambdaLR(optimizer_Enc,
                                                      lr_lambda=LambdaLR(epochs, epoch_start,
                                                                         0.2).step)
    # set the learning rate to be final LR
    for epoch in range(epoch_start):
        scheduler_Enc.step()
    for group in optimizer_Enc.param_groups:
        group["lr"] = scheduler_Enc.get_last_lr()[0]
    criterion = torch.nn.CrossEntropyLoss()

    '''
    checkpoint_path_orig_remap = f"{os.path.join(save_dir, model_name)}_starter.t7"
    state = {
        'state_dict_Enc': Enc.state_dict()}
    torch.save(state, checkpoint_path_orig_remap)
    '''
    target.eval()
    restart = True
    attempts = 0
    while restart:

        restart = False

        acc_tmp1 = pre_acc
        acc_tmp2 = pre_acc
        #acc_tmp3 = pre_acc
        #acc_tmp4 = pre_acc
        acc = pre_acc
        best = 0
        for epoch in tqdm(range(epoch_start, epochs)):
            train_hijack_transfer(target, Enc, trainset_MNIST_loader, criterion, optimizer_Enc, device)

            acc = validate_hijack_transfer(target, Enc, testset_MNIST_loader, criterion, device)
            if epoch < 8:
                if acc == acc_tmp1 and acc == acc_tmp2:# and acc == acc_tmp3 and acc == acc_tmp4:
                    restart = True
                    attempts += 1
                    if attempts > 8:
                        print("Training tranformer failed!")
                        return Enc, acc
                    epoch_start = 0
                    error_history_h = []
                    Enc.apply(weights_init_normal)
                    optimizer_Enc = torch.optim.Adam(Enc.parameters(), lr=0.01, betas=(0.5, 0.999))
                    scheduler_Enc = torch.optim.lr_scheduler.LambdaLR(optimizer_Enc,
                                                                      lr_lambda=LambdaLR(epochs, epoch_start,
                                                                                 0.2).step)
                    # set the learning rate to be final LR
                    if attempts > 3:
                        for group in optimizer_Enc.param_groups:
                            group["lr"] = 10 * group["lr"]
                    elif attempts > 5:
                        for group in optimizer_Enc.param_groups:
                            group["lr"] = 20 * group["lr"]
                    elif attempts > 7:
                        for group in optimizer_Enc.param_groups:
                            group["lr"] = 50 * group["lr"]

                    print("Terrible initialization, retrain!")
                    break
                else:
                    #acc_tmp4 = acc_tmp3
                    #acc_tmp3 = acc_tmp2
                    acc_tmp2 = acc_tmp1
                    acc_tmp1 = acc

            error_history_h.append(acc)
            scheduler_Enc.step()

            if (save_dir is not None) and (acc>best):
                best = acc
                #print('Saving...')
                state = {
                    'epoch': epoch,
                    'error_history_h': error_history_h}
                state["state_dict_Enc"] = Enc.state_dict()
                torch.save(state, checkpoint_path)

    return Enc, acc

def train_transfer_shadow(model_list, trainset_MNIST_loader, testset_MNIST_loader, epochs, model_name, save_dir, device):
    '''
    seed = 5
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    '''
    Enc = GeneratorResNet(input_channels=3, num_residual_blocks=0)

    if torch.cuda.is_available():
        for model in model_list:
            model = model.cuda()
        Enc = Enc.cuda()
        if torch.cuda.device_count() > 1:
            for model in model_list:
                model = torch.nn.DataParallel(model)
            Enc = torch.nn.DataParallel(Enc)

    if save_dir is not None:
        save_dir = os.path.join(save_dir, "transformer")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # model_name = "Enc_" + str(epoch)
        checkpoint_path = f"{os.path.join(save_dir, model_name)}.t7"

        epoch_start = 0
        error_history_h = []
        if os.path.isfile(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu")

            old_enc = state_dict["state_dict_Enc"]
            new_enc = Enc.state_dict()
            new_names = [v for v in new_enc]
            old_names = [v for v in old_enc]
            for i, j in enumerate(new_names):
                new_enc[j] = old_enc[old_names[i]]
            Enc.load_state_dict(new_enc)

            error_history_h = state_dict["error_history_h"]
            epoch_start = state_dict["epoch"] + 1
            pre_acc = error_history_h[-1]
        else:
            #Enc.apply(weights_init_normal)
            pre_acc = 0
            checkpoint_path_orig_remap_starter = '/data/gehan/CA_Hijack/output/ResNet18_CIFAR10_32_bb_t/checkpoints/transformer/Enc_orig_remap_starter2614.t7'
            if os.path.isfile(checkpoint_path_orig_remap_starter):
                state_dict = torch.load(checkpoint_path_orig_remap_starter, map_location="cpu")

                old_enc = state_dict["state_dict_Enc"]
                new_enc = Enc.state_dict()
                new_names = [v for v in new_enc]
                old_names = [v for v in old_enc]
                for i, j in enumerate(new_names):
                    new_enc[j] = old_enc[old_names[i]]
                Enc.load_state_dict(new_enc)
            else:
                Enc.apply(weights_init_normal)
    else:
        epoch_start = 0
        error_history_h = []
        pre_acc = 0
        Enc.apply(weights_init_normal)

    optimizer_Enc = torch.optim.Adam(Enc.parameters(), lr=0.01, betas=(0.5, 0.999))
    scheduler_Enc = torch.optim.lr_scheduler.LambdaLR(optimizer_Enc,
                                                      lr_lambda=LambdaLR(epochs, epoch_start,
                                                                         0.2).step)
    # set the learning rate to be final LR
    for epoch in range(epoch_start):
        scheduler_Enc.step()
    for group in optimizer_Enc.param_groups:
        group["lr"] = scheduler_Enc.get_last_lr()[0]
    criterion = torch.nn.CrossEntropyLoss()

    '''
    checkpoint_path_orig_remap = f"{os.path.join(save_dir, model_name)}_starter.t7"
    state = {
        'state_dict_Enc': Enc.state_dict()}
    torch.save(state, checkpoint_path_orig_remap)
    '''
    for model in model_list:
        model.eval()
    restart = True
    attempts = 0
    while restart:

        restart = False

        acc_tmp1 = pre_acc
        acc_tmp2 = pre_acc
        #acc_tmp3 = pre_acc
        #acc_tmp4 = pre_acc
        acc = pre_acc
        best = 0
        for epoch in tqdm(range(epoch_start, epochs)):
            Enc.train()
            for i, data_h in enumerate(trainset_MNIST_loader):
                optimizer_Enc.zero_grad()

                input_h = data_h[0].to(device)
                target_h = data_h[1].to(device)

                input_t = Enc(input_h)
                outputs = []
                for model in model_list:
                    output_h = model(input_t)
                    outputs.append(output_h)
                loss_train = 0
                for output in outputs:
                    loss_train += criterion(output, target_h)

                # compute gradient and do SGD step
                loss_train.backward()
                optimizer_Enc.step()
            acc = validate_hijack_transfer(model_list[0], Enc, testset_MNIST_loader, criterion, device)
            if epoch < 8:
                if acc == acc_tmp1 and acc == acc_tmp2:# and acc == acc_tmp3 and acc == acc_tmp4:
                    restart = True
                    attempts += 1
                    if attempts > 8:
                        print("Training tranformer failed!")
                        return Enc, acc
                    epoch_start = 0
                    error_history_h = []
                    Enc.apply(weights_init_normal)
                    optimizer_Enc = torch.optim.Adam(Enc.parameters(), lr=0.01, betas=(0.5, 0.999))
                    scheduler_Enc = torch.optim.lr_scheduler.LambdaLR(optimizer_Enc,
                                                                      lr_lambda=LambdaLR(epochs, epoch_start,
                                                                                 0.2).step)
                    # set the learning rate to be final LR
                    if attempts > 3:
                        for group in optimizer_Enc.param_groups:
                            group["lr"] = 10 * group["lr"]
                    elif attempts > 5:
                        for group in optimizer_Enc.param_groups:
                            group["lr"] = 20 * group["lr"]
                    elif attempts > 7:
                        for group in optimizer_Enc.param_groups:
                            group["lr"] = 50 * group["lr"]

                    print("Terrible initialization, retrain!")
                    break
                else:
                    #acc_tmp4 = acc_tmp3
                    #acc_tmp3 = acc_tmp2
                    acc_tmp2 = acc_tmp1
                    acc_tmp1 = acc

            error_history_h.append(acc)
            scheduler_Enc.step()

            if (save_dir is not None) and (acc>best):
                best = acc
                #print('Saving...')
                state = {
                    'epoch': epoch,
                    'error_history_h': error_history_h}
                state["state_dict_Enc"] = Enc.state_dict()
                torch.save(state, checkpoint_path)

    return Enc, acc

def train_transfer_shadow_Chameleon(model_list, trainset_MNIST_loader, testset_MNIST_loader, epochs, lr,
                                    alpha, enc_num_residual_blocks, enc_num_updownsampling,
                                    model_name, save_dir, device):
    '''
    seed = 5
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    '''
    Enc = GeneratorResNet(input_channels=3, num_residual_blocks=enc_num_residual_blocks, num_updownsampling=enc_num_updownsampling)

    if torch.cuda.is_available():
        for model in model_list:
            model = model.cuda()
        Enc = Enc.cuda()
        if torch.cuda.device_count() > 1:
            for model in model_list:
                model = torch.nn.DataParallel(model)
            Enc = torch.nn.DataParallel(Enc)

    if save_dir is not None:
        save_dir = os.path.join(save_dir, "transformer")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # model_name = "Enc_" + str(epoch)
        checkpoint_path = f"{os.path.join(save_dir, model_name)}.t7"

        epoch_start = 0
        error_history_h = []
        if os.path.isfile(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu")

            old_enc = state_dict["state_dict_Enc"]
            new_enc = Enc.state_dict()
            new_names = [v for v in new_enc]
            old_names = [v for v in old_enc]
            for i, j in enumerate(new_names):
                new_enc[j] = old_enc[old_names[i]]
            Enc.load_state_dict(new_enc)

            error_history_h = state_dict["error_history_h"]
            epoch_start = state_dict["epoch"] + 1
            pre_acc = error_history_h[-1]
        else:
            #Enc.apply(weights_init_normal)
            pre_acc = 0
            checkpoint_path_orig_remap_starter = '/data/gehan/CA_Hijack/output/ResNet18_CIFAR10_32_bb_t/checkpoints/transformer/Enc_orig_remap_starter2614.t7'
            if os.path.isfile(checkpoint_path_orig_remap_starter):
                state_dict = torch.load(checkpoint_path_orig_remap_starter, map_location="cpu")

                old_enc = state_dict["state_dict_Enc"]
                new_enc = Enc.state_dict()
                new_names = [v for v in new_enc]
                old_names = [v for v in old_enc]
                for i, j in enumerate(new_names):
                    new_enc[j] = old_enc[old_names[i]]
                Enc.load_state_dict(new_enc)
            else:
                Enc.apply(weights_init_normal)
    else:
        epoch_start = 0
        error_history_h = []
        pre_acc = 0
        Enc.apply(weights_init_normal)

    optimizer_Enc = torch.optim.Adam(Enc.parameters(), lr=lr, betas=(0.5, 0.999))
    scheduler_Enc = torch.optim.lr_scheduler.LambdaLR(optimizer_Enc,
                                                      lr_lambda=LambdaLR(epochs, epoch_start,
                                                                         0.2).step)
    # set the learning rate to be final LR
    for epoch in range(epoch_start):
        scheduler_Enc.step()
    for group in optimizer_Enc.param_groups:
        group["lr"] = scheduler_Enc.get_last_lr()[0]
    criterion = torch.nn.CrossEntropyLoss()
    criterion_dis = torch.nn.MSELoss()

    '''
    checkpoint_path_orig_remap = f"{os.path.join(save_dir, model_name)}_starter.t7"
    state = {
        'state_dict_Enc': Enc.state_dict()}
    torch.save(state, checkpoint_path_orig_remap)
    '''
    for model in model_list:
        model.eval()
    restart = True
    attempts = 0
    import copy
    while restart:

        restart = False

        acc_tmp1 = pre_acc
        acc_tmp2 = pre_acc
        #acc_tmp3 = pre_acc
        #acc_tmp4 = pre_acc
        acc = pre_acc
        best = 0
        Enc_best = copy.deepcopy(Enc)
        for epoch in tqdm(range(epoch_start, epochs)):
            Enc.train()
            for i, data_h in enumerate(trainset_MNIST_loader):
                optimizer_Enc.zero_grad()

                input_h = data_h[0].to(device)
                target_h = data_h[1].to(device)

                input_t = Enc(input_h)
                outputs = []
                for model in model_list:
                    output_h = model(input_t)
                    outputs.append(output_h)
                loss_class = 0
                loss_dis = 0
                for output in outputs:
                    loss_class += criterion(output, target_h)
                    loss_dis += criterion_dis(input_t, input_h)
                loss = loss_class + alpha*loss_dis

                # compute gradient and do SGD step
                loss.backward()
                optimizer_Enc.step()
            acc = validate_hijack_transfer(model_list[0], Enc, testset_MNIST_loader, criterion, device)
            if (epoch+1) % 5 ==0:
                torchvision.utils.save_image(input_h, '%s_origin_e%d.jpg' % (os.path.join(save_dir, model_name), epoch), nrow=8)
                torchvision.utils.save_image(input_t, '%s_transfered_e%d.jpg' % (os.path.join(save_dir, model_name), epoch), nrow=8)
            if epoch < 8:
                if acc == acc_tmp1 and acc == acc_tmp2:# and acc == acc_tmp3 and acc == acc_tmp4:
                    restart = True
                    attempts += 1
                    if attempts > 8:
                        print("Training tranformer failed!")
                        return Enc, acc
                    epoch_start = 0
                    error_history_h = []
                    Enc.apply(weights_init_normal)
                    optimizer_Enc = torch.optim.Adam(Enc.parameters(), lr=lr, betas=(0.5, 0.999))
                    scheduler_Enc = torch.optim.lr_scheduler.LambdaLR(optimizer_Enc,
                                                                      lr_lambda=LambdaLR(epochs, epoch_start,
                                                                                 0.2).step)
                    # set the learning rate to be final LR
                    if attempts > 3:
                        for group in optimizer_Enc.param_groups:
                            group["lr"] = 10 * group["lr"]
                    elif attempts > 5:
                        for group in optimizer_Enc.param_groups:
                            group["lr"] = 20 * group["lr"]
                    elif attempts > 7:
                        for group in optimizer_Enc.param_groups:
                            group["lr"] = 50 * group["lr"]

                    print("Terrible initialization, retrain!")
                    break
                else:
                    #acc_tmp4 = acc_tmp3
                    #acc_tmp3 = acc_tmp2
                    acc_tmp2 = acc_tmp1
                    acc_tmp1 = acc

            error_history_h.append(acc)
            scheduler_Enc.step()

            if (save_dir is not None) and (acc > best):
                best = acc
                Enc_best = copy.deepcopy(Enc)
                # print('Saving...')
                state = {
                    'epoch': epoch,
                    'error_history_h': error_history_h}
                state["state_dict_Enc"] = Enc.state_dict()
                torch.save(state, checkpoint_path)

    if alpha:
        state = {
            'epoch': epoch,
            'error_history_h': error_history_h}
        state["state_dict_Enc"] = Enc.state_dict()
        torch.save(state, checkpoint_path)
        return Enc, acc
    else:
        return Enc_best, best


def prune_model_wo_finetune(
        model, input_size, criterion, set_pruner, set_pruning_algo, set_sparsity,
        set_total_iteration, set_reset_weight, set_speedup,
        ft_loader, ev_loder, tr_loader, device
):
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
        TaylorFOWeightPruner,
        LinearPruner,
        AGPPruner,
        LotteryTicketPruner,
        AutoCompressPruner,
        AMCPruner,
        MovementPruner
    )
    # pre_flops, pre_params, results = count_flops_params(model, torch.randn(input_size).to(device))
    # print(f"Model FLOPs {pre_flops/1e6:.2f}M, Params {pre_params/1e6:.2f}M")

    # Prepare for Pruning
    # named_module = list(model.named_modules())
    # modules = list(model.modules())
    # state_dict = list(model.state_dict())
    # children = list(model.children())

    last_module_name, _ = list(model.named_modules())[-1]
    operations = {}
    prefix = ''
    operations = weighted_operations(model, operations, prefix)

    ifslim = False
    op_types_limitation = None
    if set_pruner in ['linear', 'agp', 'lottery', 'AutoCompress', 'AMC']:
        if set_pruning_algo in ['l1', 'l2', 'fpgm', 'apoz', 'mean_activation', 'taylorfo', 'AMC']:
            op_types_limitation = ['Conv2d', 'Linear']
        elif set_pruning_algo in ['slim']:
            ifslim = True
            op_types_limitation = None
        elif set_pruning_algo in ['level', 'admm']:
            op_types_limitation = None
    else:
        if set_pruner in ['l1', 'l2', 'fpgm', 'apoz', 'mean_activation', 'taylorfo', 'AMC']:
            op_types_limitation = ['Conv2d', 'Linear']
        elif set_pruner in ['slim']:
            ifslim = True
            op_types_limitation = ['BatchNorm2d']
        elif set_pruner in ['level', 'admm', 'Movement']:
            op_types_limitation = None

    if op_types_limitation is not None:
        for op_type in list(operations.keys()):
            check = op_type not in op_types_limitation
            if check:
                operations.pop(op_type)
    tmp = []
    for k in operations:
        if last_module_name in operations[k]:
            i = operations[k].index(last_module_name)
            del operations[k][i]
            if len(operations[k]) ==0:
                tmp.append(k)
    for t in tmp:
        del operations[t]
    op_names = []
    for v in list(operations.values()):
        op_names.extend(v)
    '''
    if last_module_name in op_names:
        op_names.remove(last_module_name)
    '''
    # 'op_types': ["default"]
    config_list = [{'op_names': op_names,
                    'sparsity': set_sparsity}]
    '''
    else:
        config_list = [{'op_types': ["default"],
                        'sparsity': set_sparsity}]
    '''

    if ifslim:
        config_list = [{
            'total_sparsity': set_sparsity,
            'op_types': ['BatchNorm2d'],
            'max_sparsity_per_layer': 0.9
        }]

    if set_pruner == 'AMC':
        #config_list[0]['max_sparsity_per_layer'] = min(0.9, set_sparsity*1.5)
        #config_list.append({'max_sparsity_per_layer': min(0.9, set_sparsity*1.5)})

        config_list = [{'op_types': ['Conv2d'],#list(operations.keys()),
                        'total_sparsity': 0.5,
                        'max_sparsity_per_layer': min(0.95, set_sparsity*1.6)}]
    if set_pruner == 'AutoCompress':
        config_list[0]['total_sparsity'] = config_list[0]['sparsity']
        del config_list[0]['sparsity']
        #config_list = [{'op_types': ['Conv2d'], 'total_sparsity': 0.8}]

    model_optimizer = optim.SGD(model.parameters(), lr=0.1,
                                momentum=0.9, weight_decay=5e-4)

    def finetuner_(model):
        finetuner(model, train_loader=ft_loader, optimizer=model_optimizer, criterion=criterion, device=device)

    def evaluator_(model):
        return evaluator(model, test_loader=ev_loder, criterion=criterion, device=device)

    def trainer_(model, optimizer, criterion):
        model.train()
        for batch_idx, (data, target) in enumerate(tr_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx and batch_idx % 100 == 0:
                print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(data), len(tr_loader.dataset),
                    100. * batch_idx / len(tr_loader), loss.item()))

    # EvaluatorBasedPruners can build [evaluator] by [trainer, traced_optimizer, criterion],
    # which includes SlimPruner, ActivationAPoZRankPruner, ActivationMeanRankPruner, TaylorFOWeightPruner, ADMMPruner
    # I set the lr in traced_optimizer as 0 so that the model will not be changed during pruning (according to official reply in github).
    # set training_epochs/training_batches as 1, since there is no change in the model during each "training" (evaluating) epoch.

    # For scheduler pruners, I set pruning iterations as 3
    if set_pruner in ['linear', 'agp', 'lottery']:
        # if you just want to keep the final result as the best result, you can pass evaluator as None.
        # or the result with the highest score (given by evaluator) will be the best result.

        kw_args = {'pruning_algorithm': set_pruning_algo,
                   'total_iteration': set_total_iteration,
                   'evaluator': evaluator_,
                   #'finetuner': finetuner_
                   }
        if set_pruning_algo in ['slim', 'apoz', 'mean_activation', 'taylorfo', 'admm']:
            traced_optimizer = nni.trace(torch.optim.SGD)(model.parameters(), lr=0, momentum=0.9, weight_decay=5e-4)
            pruning_params = {'trainer': trainer_,
                              'traced_optimizer': traced_optimizer,
                              'criterion': criterion}
            if set_pruning_algo in ['apoz', 'mean_activation', 'taylorfo']:
                pruning_params['training_batches'] = 1
            else:
                pruning_params['training_epochs'] = 1
            if set_pruning_algo in ['admm']:
                pruning_params['iterations'] = 1

            kw_args['pruning_params'] = pruning_params

        # kw_args['speedup'] = False

        # if set_speedup or set_pruning_algo in ['slim', 'apoz', 'mean_activation']:
        # kw_args['speedup'] = True
        kw_args['dummy_input'] = torch.rand(input_size).to(device)
        # elif set_pruning_algo in ['level', 'l1', 'l2', 'taylorfo']:
        #    print('speedup should be false')
        kw_args['speedup'] = set_speedup
        #kw_args['keep_intermediate_result '] = True


        if set_pruner == 'linear':
            iterative_pruner = LinearPruner
        elif set_pruner == 'agp':
            iterative_pruner = AGPPruner
        elif set_pruner == 'lottery':
            kw_args['reset_weight'] = set_reset_weight
            iterative_pruner = LotteryTicketPruner

        pruner = iterative_pruner(model, config_list, **kw_args)

        pruner.compress()
        _, model, masks, _, _ = pruner.get_best_result()
        '''
        #nni_version = nni.version
        #if nni.version <= 2.8:
        if set_pruner == 'level':
            print("Fine-grained method does not need to speedup")
        else:
            pruner._unwrap_model()
            ModelSpeedup(model, dummy_input=torch.rand(input_size).to(device), masks_file=masks).speedup_model()
        '''
    elif set_pruner == 'AutoCompress':
        traced_optimizer = nni.trace(torch.optim.SGD)(model.parameters(), lr=0, momentum=0.9, weight_decay=5e-4)

        admm_params = {
            'trainer': trainer_,
            'traced_optimizer': traced_optimizer,
            'criterion': criterion,
            'iterations': 1,
            'training_epochs': 1
        }
        sa_params = {
            'evaluator': evaluator_,
            'pruning_algorithm': set_pruning_algo

        }
        '''
        'pruning_params':{
        'trainer': trainer_,
        'traced_optimizer': traced_optimizer,
        'criterion': criterion,
        'training_epochs': 1}
        '''
        pruner = AutoCompressPruner(model, config_list, set_total_iteration, admm_params, sa_params,
                                    dummy_input=torch.rand(input_size).to(device), speedup=set_speedup, keep_intermediate_result=True)
        pruner.compress()
        _, model, masks, _, _ = pruner.get_best_result()

    elif set_pruner == 'AMC':
        # if you just want to keep the final result as the best result, you can pass evaluator as None.
        # or the result with the highest score (given by evaluator) will be the best result.
        ddpg_params = {'hidden1': 300, 'hidden2': 300, 'lr_c': 1e-3, 'lr_a': 1e-4, 'warmup': 100, 'discount': 1.,
                       'bsize': 64,
                       'rmsize': 100, 'window_length': 1, 'tau': 0.01, 'init_delta': 0.5, 'delta_decay': 0.99,
                       'max_episode_length': 1e9, 'epsilon': 50000}
        pruner = AMCPruner(5*set_total_iteration, model, config_list, dummy_input=torch.rand(input_size).to(device),
                           pruning_algorithm=set_pruning_algo, evaluator=evaluator_, ddpg_params=ddpg_params, target='params')
        pruner.compress()
        _, model, masks, _, _ = pruner.get_best_result()
    else:
        # Start to prune and speedup
        print('\n' + '=' * 50 + ' START TO PRUNE THE BEST ACCURACY PRETRAINED MODEL ' + '=' * 50)
        if set_pruner == 'level':
            pruner = LevelPruner(model, config_list)
        elif set_pruner in ['apoz', 'mean_activation', 'slim', 'taylorfo']:
            traced_optimizer = nni.trace(torch.optim.SGD)(model.parameters(), lr=0, momentum=0.9, weight_decay=5e-4)
            if set_pruner == 'apoz':
                pruner = ActivationAPoZRankPruner(model, config_list, trainer_, traced_optimizer=traced_optimizer,
                                                  criterion=criterion, training_batches=1)
            elif set_pruner == 'slim':
                pruner = SlimPruner(model, config_list, trainer_, traced_optimizer, criterion, training_epochs=1, scale=0.0001)\
                    #, mode='global')
            elif set_pruner == 'taylorfo':
                pruner = TaylorFOWeightPruner(model, config_list, trainer_, traced_optimizer=traced_optimizer, criterion=criterion,
                                              training_batches=1)

        model, masks = pruner.compress()
        pruner.show_pruned_weights()

        if set_speedup:
            if set_pruner == 'level':
                print("Fine-grained method does not need to speedup")
            else:
                pruner._unwrap_model()
                ModelSpeedup(model, dummy_input=torch.rand(input_size).to(device), masks_file=masks).speedup_model()

    return model, masks


def datasets_for_testing_hijacked_model(model_dir, batch_size):
    if 'BM' in model_dir:

        # Prepare data
        transform_train = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(size=(32, 32), interpolation=Image.BICUBIC),
             torchvision.transforms.RandomHorizontalFlip(),
             # torchvision.transforms.Resize(size=(32, 32), interpolation=Image.BICUBIC),
             torchvision.transforms.ToTensor(),
             # torchvision.transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
             # torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
             # torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
             ])
        transform_test = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(size=(32, 32), interpolation=Image.BICUBIC),
             torchvision.transforms.ToTensor(),
             # torchvision.transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
             # torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
             # torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
             ])

        # trainset = torchvision.datasets.OxfordIIITPet(root="/data/gehan/Datasets/", split="trainval", target_types="category", transform=transform_train, download=False)
        # testset = torchvision.datasets.OxfordIIITPet(root="/data/gehan/Datasets/", split="test", target_types="category", transform=transform_test, download=False)
        trainset_base = torchvision.datasets.ImageFolder("/data/gehan/Datasets/butterflies_moths/train",
                                                         transform_train)
        testset_base = torchvision.datasets.ImageFolder("/data/gehan/Datasets/butterflies_moths/test", transform_test)

        num_classes = len(trainset_base.classes)

        trainset_base_loader = torch.utils.data.DataLoader(trainset_base, batch_size=batch_size, shuffle=True)
        testset_base_loader = torch.utils.data.DataLoader(testset_base, batch_size=batch_size, shuffle=True)

        '''
        transform_train = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(size=(64, 64)),
             torchvision.transforms.RandomCrop(64, padding=4),
             torchvision.transforms.RandomHorizontalFlip(),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
        )
        transform_test = torchvision.transforms.Compose(
             [torchvision.transforms.Resize(size=(64, 64)),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
        )
        trainset_attack = torchvision.datasets.cifar.CIFAR100(root='/data/gehan/Datasets/', train=True, transform=transform_train, download=False)
        testset_attack = torchvision.datasets.cifar.CIFAR100(root='/data/gehan/Datasets/', train=False, transform=transform_test, download=False)
        '''
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(size=(32, 32), interpolation=Image.BICUBIC),
             torchvision.transforms.ToTensor(),
             # tforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
             torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
        )
        trainset_attack = torchvision.datasets.SVHN('/data/gehan/Datasets/SVHN', download=True, transform=transform,
                                                    split='train')
        testset_attack = torchvision.datasets.SVHN('/data/gehan/Datasets/SVHN', download=True, transform=transform,
                                                   split='test')

        trainset_attack_loader = torch.utils.data.DataLoader(trainset_attack, batch_size=batch_size, shuffle=True)
        testset_attack_loader = torch.utils.data.DataLoader(testset_attack, batch_size=batch_size, shuffle=True)

    elif 'Caltech' in model_dir:
        transform = torchvision.transforms.Compose(
            [
                # Image.,
                torchvision.transforms.Resize(32),
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0) if len(x) == 1 else x),
                torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
            ]
        )

        Caltech101 = torchvision.datasets.Caltech101(root='/data/gehan/Datasets/caltech-101', transform=transform,
                                                     download=False)
        len_Caltech101 = len(Caltech101)
        index_range = list(range(0, len_Caltech101))
        # test_indices = random.sample(index_range, 1000)
        test_indices = [2201, 1033, 4179, 1931, 8117, 7364, 7737, 6219, 3439, 1537, 7993, 464, 6386, 7090, 34, 7297,
                        4363,
                        3748,
                        1674, 5200, 501, 365, 416, 150, 6245, 3548, 6915, 475, 8644, 3632, 7174, 8123, 3818, 5663, 3782,
                        3584,
                        7530, 4747, 352, 6818, 1638, 3045, 4856, 1980, 5450, 8205, 8318, 3110, 4970, 4655, 8181, 8278,
                        6444,
                        565,
                        7868, 3977, 6623, 6788, 2834, 6014, 6139, 1416, 7191, 8330, 1768, 2682, 8535, 6443, 6070, 8023,
                        484,
                        7689, 712, 5054, 6448, 2791, 2762, 8228, 3718, 201, 3268, 3803, 6626, 8417, 5633, 5788, 7522,
                        4411,
                        93,
                        6286, 8396, 2117, 8498, 3366, 6981, 919, 7882, 5975, 3274, 8269, 6773, 7945, 5845, 6789, 5670,
                        25,
                        5425,
                        7506, 458, 3761, 2903, 2961, 1500, 4182, 531, 1154, 1363, 273, 7421, 238, 4607, 4088, 4401,
                        1793,
                        3024,
                        5643, 4756, 1138, 2743, 2615, 4181, 8640, 2754, 4471, 4824, 7449, 5275, 8134, 7762, 1870, 387,
                        5111,
                        6333, 5625, 6896, 3080, 4233, 1781, 4152, 8357, 3425, 7072, 341, 3692, 292, 6509, 2399, 578,
                        2625,
                        7301,
                        8295, 6990, 3614, 8463, 7386, 3656, 8583, 502, 6470, 5263, 6984, 963, 4892, 2059, 3475, 777,
                        5019,
                        1158,
                        1252, 5084, 4880, 2592, 4134, 2136, 138, 621, 3565, 7550, 2810, 8337, 613, 6192, 3283, 5684,
                        1622,
                        3371,
                        7093, 3180, 8066, 1710, 6390, 4850, 8259, 8188, 281, 5330, 6591, 4609, 296, 2571, 3290, 5369,
                        2214,
                        5555,
                        7032, 3490, 4366, 1579, 6213, 7938, 3844, 1070, 661, 1387, 2179, 2780, 2728, 3489, 4391, 5443,
                        8288,
                        6031, 5551, 5575, 1866, 4771, 3853, 8008, 2217, 1708, 5254, 641, 6661, 1199, 6229, 2413, 2048,
                        5585,
                        1879, 6193, 1255, 3665, 1339, 4370, 5978, 4842, 1872, 7500, 4541, 1765, 749, 4845, 202, 1502,
                        6775,
                        1885,
                        655, 3078, 3926, 6897, 2654, 1893, 7387, 2742, 3955, 2604, 1684, 7128, 6197, 4817, 4151, 7815,
                        5152,
                        1640, 3401, 649, 446, 172, 5246, 7370, 6410, 5132, 6529, 1031, 1051, 5199, 7468, 1824, 4097,
                        3525,
                        7682,
                        5829, 4244, 3001, 3405, 5035, 3263, 4036, 5905, 1333, 4600, 1464, 7338, 1482, 5552, 3726, 6397,
                        5026,
                        672, 5361, 3060, 5189, 4961, 4027, 5477, 1653, 1508, 4015, 3607, 333, 3993, 6582, 1185, 1161,
                        1230,
                        162,
                        4764, 5884, 8081, 7681, 2526, 8215, 5375, 1263, 8343, 2838, 2942, 2450, 2318, 5239, 5007, 1751,
                        8427,
                        4808, 2069, 3387, 2321, 520, 5178, 3365, 2918, 4897, 7088, 2586, 795, 4051, 4138, 1055, 7318,
                        7047,
                        4099,
                        7199, 7427, 178, 6483, 5548, 4226, 7959, 399, 6826, 309, 1021, 5815, 2265, 2050, 2269, 4245,
                        4536,
                        6517,
                        6571, 2820, 1462, 3826, 7962, 122, 2909, 8662, 5197, 8206, 7181, 3698, 3905, 5127, 8111, 7845,
                        3687,
                        6754, 5520, 4509, 3595, 789, 1172, 8383, 6040, 2612, 8382, 3339, 5108, 4894, 4908, 6088, 2706,
                        7614,
                        1392, 2019, 8420, 6180, 2888, 2552, 4105, 6991, 854, 8110, 5701, 6291, 8438, 2700, 666, 8588,
                        1481,
                        4180,
                        1655, 4383, 1371, 2279, 1343, 7291, 3948, 6264, 7092, 6508, 2699, 5332, 7178, 7994, 3473, 1952,
                        7065,
                        6688, 1934, 4841, 4549, 4066, 6207, 65, 8656, 7188, 344, 504, 3968, 4266, 3385, 2832, 4665,
                        2431,
                        3284,
                        4476, 5097, 4110, 7313, 2752, 5848, 8041, 6880, 1995, 3423, 6279, 3355, 4653, 1771, 395, 216,
                        2237,
                        1231,
                        8198, 6123, 5099, 7162, 8241, 5846, 8657, 5303, 13, 2029, 7246, 7365, 5737, 4993, 6543, 5560,
                        8065,
                        1852,
                        6185, 6265, 3340, 63, 4548, 8371, 3258, 7562, 8469, 6700, 5002, 2790, 7362, 3233, 5888, 8621,
                        57,
                        6376,
                        6977, 6639, 5505, 1109, 8072, 4057, 4765, 340, 6668, 2557, 4427, 1202, 165, 5725, 4334, 6736,
                        4975,
                        2491,
                        7570, 4249, 2779, 7653, 8361, 743, 4437, 8360, 1615, 6923, 1142, 5819, 1097, 7249, 323, 2689,
                        8309,
                        2648,
                        1524, 6585, 4518, 4987, 3422, 8652, 3403, 3886, 5471, 4408, 1123, 1226, 8572, 6032, 7666, 8380,
                        814,
                        2761, 4864, 4419, 5830, 3802, 6431, 6548, 2823, 7923, 4252, 5400, 3642, 4239, 4001, 500, 6596,
                        5186,
                        7074, 4070, 3111, 1188, 2713, 7267, 2427, 4292, 7526, 8627, 2662, 2271, 2262, 7220, 5916, 5075,
                        6565,
                        3940, 1897, 3378, 5005, 1117, 1743, 3729, 6504, 5265, 1637, 3059, 736, 906, 381, 568, 8101,
                        8659,
                        5610,
                        4498, 2829, 1560, 3638, 3821, 7369, 6191, 3796, 3862, 4647, 7578, 6383, 3471, 7400, 4225, 5408,
                        8131,
                        1817, 3503, 1291, 757, 252, 85, 7870, 5235, 6277, 4705, 3209, 6552, 2622, 2494, 499, 248, 6345,
                        2378,
                        935, 6217, 4164, 2129, 1302, 7583, 236, 581, 996, 8600, 2112, 701, 4482, 1924, 7086, 1491, 3114,
                        452,
                        8186, 2135, 4575, 3144, 7332, 6384, 5403, 4390, 4257, 3982, 4021, 986, 2871, 5728, 7020, 8555,
                        5787,
                        6760, 3266, 6948, 1148, 4376, 1184, 4121, 1582, 2474, 961, 3331, 7014, 735, 865, 1494, 8402,
                        7686,
                        8210,
                        6066, 1626, 5123, 657, 2074, 543, 7263, 2100, 6474, 7308, 403, 8593, 4423, 1480, 4096, 5331,
                        1405,
                        4945,
                        560, 6295, 952, 4276, 5131, 2130, 4264, 6228, 1919, 4976, 1541, 6960, 4020, 8236, 8344, 6407,
                        7883,
                        1715,
                        2125, 7350, 8581, 8520, 495, 4773, 2572, 3276, 6067, 6377, 8537, 5312, 1595, 6709, 5658, 2070,
                        1062,
                        713,
                        4923, 5138, 6841, 4887, 5223, 5777, 4467, 5329, 8521, 8209, 141, 8620, 1996, 2437, 5195, 5334,
                        5366,
                        1127, 7402, 4581, 7859, 7440, 5966, 6234, 1280, 2204, 798, 8580, 8063, 4127, 5924, 6064, 6595,
                        5036,
                        7611, 5577, 8315, 2749, 476, 2430, 4098, 3623, 2185, 1847, 6735, 820, 1625, 4353, 1752, 3347,
                        4287,
                        1094,
                        8624, 1286, 1192, 3561, 2840, 7079, 357, 7973, 4648, 3603, 8087, 6970, 7408, 6015, 3093, 7899,
                        1191,
                        4203, 6673, 3299, 135, 6237, 8426, 7980, 1251, 6614, 8356, 6972, 5764, 7511, 104, 3109, 4904,
                        90,
                        1966,
                        4958, 5170, 4628, 8611, 6740, 8484, 6689, 5042, 7414, 4946, 2145, 7277, 2299, 2670, 4140, 157,
                        6949,
                        593,
                        6035, 6895, 6588, 4612, 300, 1475, 78, 6281, 4405, 7608, 4455, 6105, 7887, 5513, 6364, 7473,
                        1908,
                        7925,
                        5808, 2370, 6802, 2429, 297, 2819, 4263, 6025, 2082, 4704, 6765, 4706, 6893, 4483, 7102, 5503,
                        3530,
                        8050, 6584, 6965, 1497, 2121, 3377, 2451, 3755, 428, 1691, 4148, 2551, 7860, 1621, 6539, 3070,
                        49,
                        1460,
                        7007, 833, 3576, 6912, 5680, 770, 1690, 6875, 1943, 4347, 4567, 2933, 781, 3509, 1428, 6385,
                        2028,
                        7328,
                        4820, 8320, 8158, 6440, 1903, 7851, 1733, 2443, 6330, 3296, 2738, 8531, 4220, 6825, 4728, 8068,
                        3516,
                        5522, 1685, 140, 5683, 924, 7213, 4912, 1650, 3744, 8323, 4429, 6744, 2133, 4199, 3199, 6680,
                        957,
                        8345,
                        2438, 6779, 4426, 4584, 7866, 5010, 4375, 8049, 3512, 8171, 6024, 7709]

        train_indices = index_range
        for x in test_indices:
            train_indices.remove(x)
        trainset_Caltech101 = torch.utils.data.Subset(Caltech101, train_indices)
        testset_Caltech101 = torch.utils.data.Subset(Caltech101, test_indices)
        trainset_base_loader = torch.utils.data.DataLoader(trainset_Caltech101, batch_size=batch_size,
                                                           shuffle=True)
        testset_base_loader = torch.utils.data.DataLoader(testset_Caltech101, batch_size=batch_size,
                                                          shuffle=True)

        num_classes = len(Caltech101.categories)

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
        trainset_CIFAR100 = torchvision.datasets.cifar.CIFAR100(root='/data/gehan/Datasets/', train=True,
                                                                transform=transform_train, download=False)
        testset_CIFAR100 = torchvision.datasets.cifar.CIFAR100(root='/data/gehan/Datasets/', train=False,
                                                               transform=transform_test, download=False)
        trainset_attack_loader = torch.utils.data.DataLoader(trainset_CIFAR100, batch_size=batch_size,
                                                             shuffle=True)
        testset_attack_loader = torch.utils.data.DataLoader(testset_CIFAR100, batch_size=batch_size, shuffle=True)

    elif '100' in model_dir:
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

        trainset_CIFAR100 = torchvision.datasets.cifar.CIFAR100(root='/data/gehan/Datasets/', train=True,
                                                                transform=transform_train, download=False)
        testset_CIFAR100 = torchvision.datasets.cifar.CIFAR100(root='/data/gehan/Datasets/', train=False,
                                                               transform=transform_test, download=False)
        trainset_base_loader = torch.utils.data.DataLoader(trainset_CIFAR100, batch_size=batch_size, shuffle=True)
        testset_base_loader = torch.utils.data.DataLoader(testset_CIFAR100, batch_size=batch_size, shuffle=True)

        num_classes = len(trainset_CIFAR100.classes)
        '''
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(size=(32, 32), interpolation=Image.BICUBIC),
             torchvision.transforms.ToTensor(),
             # tforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
             torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
        )
        trainset_SVHN = torchvision.datasets.SVHN('/data/gehan/Datasets/SVHN', download=True, transform=transform,
                                                  split='train')
        testset_SVHN = torchvision.datasets.SVHN('/data/gehan/Datasets/SVHN', download=True, transform=transform,
                                                 split='test')
        trainset_attack_loader = torch.utils.data.DataLoader(trainset_SVHN, batch_size=batch_size, shuffle=True)
        testset_attack_loader = torch.utils.data.DataLoader(testset_SVHN, batch_size=batch_size, shuffle=True)
        
        transform_train = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(size=(32, 32), interpolation=Image.BICUBIC),
             torchvision.transforms.RandomHorizontalFlip(),
             # torchvision.transforms.Resize(size=(32, 32), interpolation=Image.BICUBIC),
             torchvision.transforms.ToTensor(),
             # torchvision.transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
             # torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
             # torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
             ])
        transform_test = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(size=(32, 32), interpolation=Image.BICUBIC),
             torchvision.transforms.ToTensor(),
             # torchvision.transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
             # torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
             # torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
             ])
        # trainset_attack = torchvision.datasets.OxfordIIITPet(root="/data/gehan/Datasets/", split="trainval", target_types="category", transform=transform_train, download=False)
        # testset_attack = torchvision.datasets.OxfordIIITPet(root="/data/gehan/Datasets/", split="test", target_types="category", transform=transform_test, download=False)

        trainset_attack = torchvision.datasets.ImageFolder("/data/gehan/Datasets/butterflies_moths/train",
                                                           transform_train)
        testset_attack = torchvision.datasets.ImageFolder("/data/gehan/Datasets/butterflies_moths/test", transform_test)
        '''

        transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(size=(32, 32), interpolation=Image.BICUBIC),
             torchvision.transforms.ToTensor(),
             # tforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
             torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
        )
        trainset_attack = torchvision.datasets.GTSRB('/data/gehan/Datasets/', split='train', transform=transform,
                                                     download=False)
        testset_attack = torchvision.datasets.GTSRB('/data/gehan/Datasets/', split='test', transform=transform,
                                                    download=False)

        trainset_attack_loader = torch.utils.data.DataLoader(trainset_attack, batch_size=batch_size, shuffle=True)
        testset_attack_loader = torch.utils.data.DataLoader(testset_attack, batch_size=batch_size, shuffle=True)

    else:
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
        trainset_base_loader = torch.utils.data.DataLoader(trainset_CIFAR10, batch_size=batch_size, shuffle=True)
        testset_base_loader = torch.utils.data.DataLoader(testset_CIFAR10, batch_size=batch_size, shuffle=True)

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
        trainset_attack_loader = torch.utils.data.DataLoader(trainset_MNIST, batch_size=batch_size, shuffle=True)
        testset_attack_loader = torch.utils.data.DataLoader(testset_MNIST, batch_size=batch_size, shuffle=True)

    return num_classes, trainset_base_loader, testset_base_loader, trainset_attack_loader, testset_attack_loader

def remap(model, dataloader, device):
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix

    print("Remapping......")
    # Calculate confusion matrix
    y_true, y_pred = [], []
    count_total, count_correct = 0, 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            count_total += labels.size(0)
            count_correct += preds.eq(labels).sum().item()
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

            if count_total >= 10000:
                break
        print('>> Test accuracy is: %.3f%%'
              % (100. * count_correct / count_total))
    # Calculate a confusion matrix to analyze the relationship between true labels and predicted labels.
    cm = confusion_matrix(y_true, y_pred)
    # Find the optimal label mapping using the Hungarian algorithm. Use the Hungarian algorithm to find the optimal label mapping based on the confusion matrix.
    row_ind, col_ind = linear_sum_assignment(-cm)
    label_mapping = dict(zip(row_ind, col_ind))
    print(label_mapping)

    return label_mapping

class suppress_stdout_stderr(object):
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])