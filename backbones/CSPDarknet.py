import torch
import torch.nn as nn
import os
import torch.nn.functional as F

__all__ = ['cspdarknet53']


def Mish(x):
    # Mish激活函数是由Swish函数受启发而来
    return x * (torch.tanh(F.softplus(x)))


# 构建CBM块
# Conv + BN + Mish
class CBM(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, act=True):
        super(CBM, self).__init__()
        self.act = act
        if self.act:
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p,
                          dilation=d, groups=g, bias=False),
                nn.BatchNorm2d(c2),
            )
        else:  # 残差连接中后面的一层卷积，没有激活函数
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p,
                          dilation=d, groups=g, bias=False),
                nn.BatchNorm2d(c2)
            )

    def forward(self, x):
        if self.act:
            return Mish(self.convs(x))
        else:  # 残差连接中后面的一层卷积，没有激活函数
            return self.convs(x)


class ResidualBlock(nn.Module):
    """
    basic residual block for CSP-Darknet
    """

    def __init__(self, in_ch):
        super(ResidualBlock, self).__init__()
        self.conv1 = CBM(in_ch, in_ch, k=1)
        self.conv2 = CBM(in_ch, in_ch, k=3, p=1, act=False)
        # self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        h = self.conv2(self.conv1(x))

        # Resiual中的add连接
        return Mish(x+h)
        # out = self.act(x + h)
        # return out


class CSPStage(nn.Module):
    def __init__(self, c1, n=1):
        super(CSPStage, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = CBM(c1, c_, k=1)
        self.cv2 = CBM(c1, c_, k=1)
        # 每个block重复的层数为：1 2 8 8 4
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(in_ch=c_) for _ in range(n)])
        self.cv3 = CBM(2 * c_, c1, k=1)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.res_blocks(self.cv2(x))

        # CSP中的Concat连接
        return self.cv3(torch.cat([y1, y2], dim=1))


# CSPDarknet
class CSPDarknet53(nn.Module):
    """
    CSPDarknet_53.
    """

    def __init__(self, num_classes=10):
        super(CSPDarknet53, self).__init__()

        self.layer_1 = nn.Sequential(
            CBM(3, 32, k=3, p=1),
            CBM(32, 64, k=3, p=1, s=2),
            CSPStage(c1=64, n=1)  # p1/2
        )
        self.layer_2 = nn.Sequential(
            CBM(64, 128, k=3, p=1, s=2),
            CSPStage(c1=128, n=2)  # P2/4
        )
        self.layer_3 = nn.Sequential(
            CBM(128, 256, k=3, p=1, s=2),
            CSPStage(c1=256, n=8)  # P3/8
        )
        self.layer_4 = nn.Sequential(
            CBM(256, 512, k=3, p=1, s=2),
            CSPStage(c1=512, n=8)  # P4/16
        )
        self.layer_5 = nn.Sequential(
            CBM(512, 1024, k=3, p=1, s=2),
            CSPStage(c1=1024, n=4)  # P5/32
        )

        # 最后两层后处理
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)
        # return c3, c4, c5

        # 为了适配cifar10，加上全局均值池化核全连接层
        out = self.avgpool(c5)
        out = c5.view(out.size(0), -1)
        out = self.fc(out)
        return out


# CSPDarkNet-Tiny
class CSPDarknetTiny(nn.Module):
    """
    CSPDarknet_Tiny.
    """

    def __init__(self):
        super(CSPDarknetTiny, self).__init__()

        self.layer_1 = nn.Sequential(
            CBM(3, 16, k=3, p=1),
            CBM(16, 32, k=3, p=1, s=2),
            CSPStage(c1=32, n=1)  # p1/2
        )
        self.layer_2 = nn.Sequential(
            CBM(32, 64, k=3, p=1, s=2),
            CSPStage(c1=64, n=1)  # P2/4
        )
        self.layer_3 = nn.Sequential(
            CBM(64, 128, k=3, p=1, s=2),
            CSPStage(c1=128, n=1)  # P3/8
        )
        self.layer_4 = nn.Sequential(
            CBM(128, 256, k=3, p=1, s=2),
            CSPStage(c1=256, n=1)  # P4/16
        )
        self.layer_5 = nn.Sequential(
            CBM(256, 512, k=3, p=1, s=2),
            CSPStage(c1=512, n=1)  # P5/32
        )

    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        print(c4.shape)
        c5 = self.layer_5(c4)
        print(c5.shape)
        # return c3, c4, c5
        return c5


def cspdarknet53(pretrained=False, **kwargs):
    """Constructs a CSPDarknet53 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = CSPDarknet53()
    if pretrained:
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        print('Loading the cspdarknet53 ...')
        model.load_state_dict(torch.load(
            path_to_dir + '/weights/cspdarknet53/cspdarknet53.pth'), strict=False)
    return model


# net = cspdarknet53()
# x = torch.randn(4, 3, 32, 32)
# y = net(x)
# print(y.shape)
