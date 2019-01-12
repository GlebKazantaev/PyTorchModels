import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = BasicBlock.conv3x3(inplanes, outplanes, stride)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = BasicBlock.conv3x3(outplanes, outplanes)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)

        out += residual
        #out = self.relu(out)

        return out

    @staticmethod
    def conv3x3(in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=False)


class DeblurResnetModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2, bias=False)
        #self.bn1 = nn.BatchNorm2d(128)

        self.layer1 = self._make_layer()
        self.layer2 = self._make_layer()
        self.layer3 = self._make_layer()
        self.layer4 = self._make_layer()
        self.layer5 = self._make_layer()
        self.layer6 = self._make_layer()
        self.layer7 = self._make_layer()
        self.layer8 = self._make_layer()
        self.layer9 = self._make_layer()
        self.layer10 = self._make_layer()
        self.layer11 = self._make_layer()
        self.layer12 = self._make_layer()
        self.layer13 = self._make_layer()
        self.layer14 = self._make_layer()
        self.layer15 = self._make_layer()
        self.layer16 = self._make_layer()
        self.layer17 = self._make_layer()
        self.layer18 = self._make_layer()
        self.layer19 = self._make_layer()


        self.conv2 = nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self):
        layers = []
        layers.append(BasicBlock(64, 64))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = self.layer18(x)
        x = self.layer19(x)
        x = self.conv2(x)

        return x
