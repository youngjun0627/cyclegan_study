# ref: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

import functools
import torch.nn as nn
import torch
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, model_name='ResNetFPN', n_blocks=6):
        super(Generator, self).__init__()
        if model_name == 'ResNet':
            self.model = ResNet(input_nc, output_nc)
        elif model_name == 'ResNetFPN':
            self.model = ResNetFPN(input_nc, output_nc, ResNetFPNBlock)
        
    def forward(self, x):
        return self.model(x)


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

            model += [ResnetBlock(ngf * mult, use_dropout=use_dropout)]

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

class ResNetFPN(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        block,
        ngf=64,
        use_dropout=True,
        layers=[3, 4, 6, 3],
        fpn_weights = [1., 1., 1., 1.]
    ):
        # ResNet(input_nc, output_nc, ngf, fpn_weights, BasicBlock_Ganilla, [3, 4, 6, 3], use_dropout=use_dropout, **kwargs)
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
        self.layer1 = self._make_layer(block, 64, layers[0], use_dropout, stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], use_dropout, stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], use_dropout, stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], use_dropout, stride=2)
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

    def _make_layer(self, block, planes, blocks, use_dropout, stride=1):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, use_dropout, stride))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        out = self.fpn([x1, x2, x3, x4]) # use all resnet layers
        out = self.layer5(out)

        return out




class ResNetFPNBlock(nn.Module):
    def __init__(self, in_planes, planes, use_dropout, stride=1):
        super(ResNetFPNBlock, self).__init__()
        self.rp1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=0, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(0.5)

        self.rp2 = nn.ReflectionPad2d(1)
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
        out = F.relu(self.bn1(self.conv1(self.rp1(x))))
        if self.use_dropout:
            out = self.dropout(out)
        out = self.bn2(self.conv2(self.rp2(out)))
        residual_input = self.shortcut(x)
        concat_out = torch.cat((out, residual_input), 1)
        out = self.final_conv(concat_out)
        out = F.relu(out)
        return out

class PyramidFeatures(nn.Module):
    def __init__(self, C2_size, C3_size, C4_size, C5_size, fpn_weights, feature_size=128):
        super(PyramidFeatures, self).__init__()

        self.sum_weights = fpn_weights #[1.0, 0.5, 0.5, 0.5]

        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.rp4 = nn.ReflectionPad2d(1)
        self.P2_2 = nn.Conv2d(int(feature_size), int(feature_size/2), kernel_size=3, stride=1, padding=0)


    def forward(self, inputs):

        C2, C3, C4, C5 = inputs

        i = 0
        P5_x = self.P5_1(C5) * self.sum_weights[i]
        P5_upsampled_x = self.P5_upsampled(P5_x)
        
        i += 1
        P4_x = self.P4_1(C4) * self.sum_weights[i]
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        
        i += 1
        P3_x = self.P3_1(C3) * self.sum_weights[i]
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        
        i += 1
        P2_x = self.P2_1(C2) * self.sum_weights[i]
        P2_x = P2_x * self.sum_weights[2] + P3_upsampled_x
        P2_upsampled_x = self.P2_upsampled(P2_x)
        P2_x = self.rp4(P2_upsampled_x)
        P2_x = self.P2_2(P2_x)

        return P2_x


if __name__ == '__main__':
    model = Generator(3,3, use_dropout=True)
    #model = ResNetFPN(3, 3, 64, ResNetBlock)
    print(model)
    print(model(torch.rand(2,3,256,256)).shape)
