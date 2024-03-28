'''AlexNet in PyTorch.'''
# 原论文中的AlexNet的网络结构
# 不适用于进行cifar10分类
# 因为cifar10图像大小是32 * 32，经过下采样后，输出tensor为0
# 因此，这里为了适配cifar10数据集，而进行适当改进
# 将特征提取层最后的最大池化层，更改为均值池化层
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        # 特征提取层
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11,
                      stride=4, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=96, out_channels=192,
                      kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=192, out_channels=384,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 分类卷积层
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
