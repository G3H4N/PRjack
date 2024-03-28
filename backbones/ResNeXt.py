'''ResNeXt in PyTorch.

See the paper "Aggregated Residual Transformations for Deep Neural Networks" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    # 每个block中第三层相对于前两层的channel扩展的倍数
    planes_expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width

        # 1 × 1， C=cardinality
        self.conv1 = nn.Conv2d(in_planes, group_width,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)

        # 3 × 3， C=cardinality
        # groups: 默认为1，从输入通道到输出通道的连接块数，对输入输出通道进行分组分为groups组
        # groups=2时，一次卷积操作的效果相当于两层卷积层并排在一起，每一层负责输入通道的一半，负责输出通道的一半
        # 然后将两部分输出concatenate到一起；out_channels=in_channels，就是所谓的逐通道卷积（depthwise convolution）
        # 每个通道的卷积核数为输出通道数与输入通道数的比
        # 因此，这里out_channels=in_channels=cardinality * bottleneck_width
        # 分为cardinality个group，每个group的channel=bottleneck_width
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3,
                               stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)

        # 1 × 1， C=cardinality
        self.conv3 = nn.Conv2d(
            group_width, self.planes_expansion * group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.planes_expansion * group_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.planes_expansion * group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.planes_expansion * group_width,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.planes_expansion * group_width)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=10):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        # 如果是resnext29，num_blocks=[3, 3, 3]
        # 那么第四层就不要了
        # self.layer4 = self._make_layer(num_blocks[3], 2)

        # 如果是resnext29，num_blocks=[3, 3, 3]
        self.linear = nn.Linear(cardinality*bottleneck_width*8, num_classes)
        # self.linear = nn.Linear(cardinality * bottleneck_width * 16, num_classes)

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality,
                                self.bottleneck_width, stride))
            self.in_planes = Block.planes_expansion * \
                             self.cardinality * self.bottleneck_width

        # Increase bottleneck_width by 2 after each stage.
        # 每个block之间，channel数增加2倍
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # 如果是resnext29，num_blocks=[3, 3, 3]
        # 那么第四层就不要了
        # out = self.layer4(out)

        # 如果是resnext29，num_blocks=[3, 3, 3]
        out = F.avg_pool2d(out, 8)
        # out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNeXt29_2x64d():
    return ResNeXt(num_blocks=[3, 3, 3], cardinality=2, bottleneck_width=64)


def ResNeXt29_4x64d():
    return ResNeXt(num_blocks=[3, 3, 3], cardinality=4, bottleneck_width=64)


def ResNeXt29_8x64d():
    return ResNeXt(num_blocks=[3, 3, 3], cardinality=8, bottleneck_width=64)


def ResNeXt29_32x4d():
    return ResNeXt(num_blocks=[3, 3, 3], cardinality=32, bottleneck_width=4)


def ResNeXt50_32x4d():
    return ResNeXt(num_blocks=[3, 4, 6, 3], cardinality=32, bottleneck_width=4)


def ResNeXt101_32x4d():
    return ResNeXt(num_blocks=[3, 4, 23, 3], cardinality=32, bottleneck_width=4)
