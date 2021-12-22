# ref: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super(Discriminator, self).__init__()
        kw = 4
        padw = 1
        model = [nn.Conv2d(input_nc, ndf, kernel_size=kw,stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers + 1):
            nf_mult_prev = nf_mult
            nf_mult = 2 ** n
            if n == n_layers:
                model += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
                    nn.InstanceNorm2d(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                model += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                    nn.InstanceNorm2d(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = Discriminator(3)
    print(model)
