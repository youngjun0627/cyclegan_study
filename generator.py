# ref: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

import torch.nn as nn
import torch
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, model_name='ResNetFPN', n_blocks=6):
        super(Generator, self).__init__()
        if model_name == 'ResNet':
            self.model = ResNet(input_nc, output_nc)
        elif model_name == 'ResNetFPN':
            self.model = ResNetFPN(input_nc, output_nc)

    def forward(self, x):
        return self.model(x)


# Resnet
class ResNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, n_blocks=6):
        super(ResNet, self).__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResNetBlock(ngf * mult, use_dropout=use_dropout)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1,
                                   ),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResNetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, use_dropout):
        super(ResNetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, use_dropout)

    def build_conv_block(self, dim, use_dropout):
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True)
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


#  Resnet Feature Pyramid Network
class ResNetFPN(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        use_dropout=True,
        layers=[3, 4, 6, 3],
        fpn_weights=[1.0, 0.5, 0.5, 0.5]
    ):
        super(ResNetFPN, self).__init__()
        self.inplanes = ngf
        self.layer0 = nn.Sequential(
            nn.ReflectionPad2d(input_nc),
            nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        )
        self.layer1 = self._make_layer(64, layers[0], use_dropout, stride=1)
        self.layer2 = self._make_layer(128, layers[1], use_dropout, stride=2)
        self.layer3 = self._make_layer(128, layers[2], use_dropout, stride=2)
        self.layer4 = self._make_layer(256, layers[3], use_dropout, stride=2)
        self.layer5 = nn.Sequential(
            nn.ReflectionPad2d(output_nc),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()
        )
        self.fpn = PyramidFeatures(
            64,
            128,
            128,
            256,
            fpn_weights
        )

    def _make_layer(self, planes, blocks, use_dropout, stride=1):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResNetFPNBlock(self.inplanes, planes, use_dropout, stride))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        out = self.fpn([x4, x3, x2, x1])
        out = self.layer5(out)

        return out


class ResNetFPNBlock(nn.Module):
    def __init__(self, in_planes, planes, use_dropout, stride=1):
        super(ResNetFPNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=0, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(0.5)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.InstanceNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(planes)
            )

        self.final_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(planes * 2, planes, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(planes)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(self.pad1(x))))
        if self.use_dropout:
            out = self.dropout(out)
        out = self.bn2(self.conv2(self.pad2(out)))
        residual_input = self.shortcut(x)
        concat_out = torch.cat((out, residual_input), 1)
        out = self.final_conv(concat_out)
        out = F.relu(out)
        return out


class PyramidFeatures(nn.Module):
    def __init__(self, F2_size, F3_size, F4_size, F5_size, fpn_weights, output_size=128):
        super(PyramidFeatures, self).__init__()

        self.weights = fpn_weights

        self.convList = nn.ModuleList([
            nn.Conv2d(
                feature_size,
                output_size,
                kernel_size=1,
                stride=1,
                padding=0
            ) for feature_size in [F5_size, F4_size, F3_size, F2_size]
        ])

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.last_pad = nn.ReflectionPad2d(1)
        self.last_conv = nn.Conv2d(output_size, output_size // 2, kernel_size=3, stride=1, padding=0)

    def forward(self, inputs):

        output = None
        for i, (x, conv) in enumerate(zip(inputs, self.convList)):
            x = conv(x) * self.weights[i]
            if output is None:
                output = self.upsample(x)
            else:
                output = self.upsample(output + x)
        output = self.last_pad(output)
        output = self.last_conv(output)
        return output


if __name__ == '__main__':
    model = Generator(3, 3, use_dropout=True)
    #model = ResNetFPN(3, 3, 64, ResNetBlock)
    print(model)
    print(model(torch.rand(2, 3, 256, 256)).shape)
