"""
pytorch实现ResNet50、ResNet101和ResNet152:
https://www.jianshu.com/p/0df349249b12
"""
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

print('torch_version: ', torch.__version__)
print('torchvision_version: ', torchvision.__version__)


# conv1 7 x 7 64 stride=2
def Conv1(channel_in, channel_out, stride=2):
    return nn.Sequential(
        nn.Conv2d(
            channel_in,
            channel_out,
            kernel_size=7,
            stride=stride,
            padding=3,
            bias=False
        ),
        nn.BatchNorm2d(channel_out),
        # 会改变输入数据的值
        # 节省反复申请与释放内存的空间与时间
        # 只是将原来的地址传递，效率更好
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
    )


# 构建ResNet18-34的网络基础模块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 构建ResNet50-101-152的网络基础模块
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        # 构建 1x1, 3x3, 1x1的核心卷积块
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        # 采用1x1的kernel，构建shout cut
        # 注意这里除了第一个bottleblock之外，都需要下采样，所以步长要设置为stride=2
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 搭建ResNet模板块
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        #CHANGE！！！！
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 逐层搭建ResNet
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)


        #self.last_conv = torch.nn.Conv2d(in_channels=self.linear.in_features, out_channels=num_classes, kernel_size=1)

        # Xavier参数初始化
        # 基本思想：保持输入和输出的方差一致
        # 这样就避免了所有输出值都趋向于0
        # 这是通用的方法，适用于任何激活函数
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        # layers = [ ] 是一个列表
        # 通过下面的for循环遍历配置列表，可以得到一个由 卷积操作、池化操作等 组成的一个列表layers
        # return nn.Sequential(*layers)，即通过nn.Sequential函数将列表通过非关键字参数的形式传入(列表layers前有一个星号)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        #CHANGE！！！！
        #out = F.avg_pool2d(out, 4)
        out = self.avgpool(out)

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        '''
        out = F.avg_pool2d(out, out.size()[2])
        out = self.last_conv(out)
        out = torch.squeeze(out, -1)
        out = torch.squeeze(out, -1)
        '''
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

# 测试
# if __name__ == '__main__':
#     model = ResNet50()
#     print(model)
#
#     input = torch.randn(1, 3, 32, 32)
#     out = model(input)
#     print(out.shape)
