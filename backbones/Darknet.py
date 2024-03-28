import torch
import torch.nn as nn
import os

__all__ = ['darknet53']


# CBL层
# Conv2d + BN + Leak_ReLU
class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.CBL = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize,
                      padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.CBL(x)


# 1 × 1、3 × 3、shortcut
class ResBlock(nn.Module):
    def __init__(self, ch, nblocks=1):
        super().__init__()
        self.module_list = nn.ModuleList()
        # 每个block重复次数：1 2 8 8 4
        for _ in range(nblocks):
            resblock_one = nn.Sequential(
                Conv_BN_LeakyReLU(ch, ch // 2, 1),
                Conv_BN_LeakyReLU(ch // 2, ch, 3, padding=1)
            )
            self.module_list.append(resblock_one)

    def forward(self, x):
        # 实现residual，残差跳连
        for module in self.module_list:
            x = module(x) + x
        return x


class DarkNet53(nn.Module):
    """
    DarkNet-53.
    """

    def __init__(self, num_classes=10):
        super(DarkNet53, self).__init__()
        # stride = 2
        self.layer_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, padding=1),
            Conv_BN_LeakyReLU(32, 64, 3, padding=1, stride=2),
            ResBlock(64, nblocks=1)
        )
        # stride = 4
        self.layer_2 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, padding=1, stride=2),
            ResBlock(128, nblocks=2)
        )
        # stride = 8
        self.layer_3 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, padding=1, stride=2),
            ResBlock(256, nblocks=8)
        )
        # stride = 16
        self.layer_4 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, padding=1, stride=2),
            ResBlock(512, nblocks=8)
        )
        # stride = 32
        self.layer_5 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, padding=1, stride=2),
            ResBlock(1024, nblocks=4)
        )

        # 最后两层后处理
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x, targets=None):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        # 在YOLOv3中返回后三层feature map
        # return c3, c4, c5

        # 为了适配cifar10，加上全局均值池化核全连接层
        out = self.avgpool(c5)
        out = c5.view(out.size(0), -1)
        out = self.fc(out)
        return out


def darknet53(pretrained=False, **kwargs):
    """Constructs a darknet-53 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DarkNet53()
    if pretrained:
        print('Loading the pretrained model ...')
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        print('Loading the darknet53 ...')
        model.load_state_dict(torch.load(
            path_to_dir + '/weights/darknet53/darknet53.pth'), strict=False)
    return model


# net = darknet53()
# # print(net)
# y = net(torch.rand((4, 3, 32, 32)))
#
# print(y.shape)
