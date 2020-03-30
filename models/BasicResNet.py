import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        residual = self.shortcut(residual)

        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        return self.relu(x)


class resnet(nn.Module):
    def __init__(self, num_classes, num_block_lists=[2, 2, 2, 2]):
        super(resnet, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.stage_1 = self._make_layer(64, 64, nums_block=num_block_lists[0], stride=1)
        self.stage_2 = self._make_layer(64, 128, nums_block=num_block_lists[1], stride=2)
        self.stage_3 = self._make_layer(128, 256, nums_block=num_block_lists[2], stride=2)
        self.stage_4 = self._make_layer(256, 512, nums_block=num_block_lists[3], stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, in_channels, out_channels, nums_block, stride=1):
        layers = [BasicBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, nums_block):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.basic_conv(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x


def ResNet(num_classes=1000, depth=18):
    assert depth in [18, 34], 'depth invalid'
    key2blocks = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
    }
    model = resnet(num_classes, key2blocks[depth])
    return model


if __name__ == '__main__':
    batch = 2
    inplanes = 3
    outplanes = 1000
    h, w = 224, 224

    model = ResNet(outplanes, depth=18)
    x = torch.rand((batch, inplanes, h, w))
    model(x)
    print(model)
