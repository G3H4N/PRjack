'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        # PW 扩张：逐点卷积
        # 使得深度卷积在相对更深的维度上进行特征提取
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # DW Conv：深度卷积，将卷积核拆分成为单通道形式，在不改变输入特征图像深度的情况下，对每一通道进行卷积操作
        # 因此，groups=planes
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # PW 压缩：逐点卷积
        # 就是1×1卷积，主要作用就是对特征图进行升维和降维
        self.conv3 = nn.Conv2d(
            planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetv2(nn.Module):
    # 论文中的网络参数
    # (expansion: t, out_planes: c, num_blocks: n, stride: s)
    cfg = [(1, 16, 1, 1),
           (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6, 32, 3, 2),
           (6, 64, 4, 2),
           (6, 96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetv2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # 搭建MobileNetV2的网络架构
        self.layers = self._make_layers(in_planes=32)

        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        # 遍历MobileNetV2网络中的每一block
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)  # 后面跟着重复的1
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                # 后一层输入的channel数，就是当前层的channel数
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        #CHANGE!!!
        #out = F.avg_pool2d(out, 4)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# def MobileNetv2():
#     return MobileNetv2_Main()
# def test():
#     net = MobileNetv2()
#     x = torch.randn(2, 3, 32, 32)
#     y = net(x)
#     print(y.size())


# test()
