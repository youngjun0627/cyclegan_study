# ref: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

import functools
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, n_blocks=6):
        super(Generator, self).__init__()
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


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, use_dropout):
        super(ResnetBlock, self).__init__()
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

if __name__ == '__main__':
    model = Generator(3,3)
    print(model)
