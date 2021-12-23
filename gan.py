# ref: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

import torch
import torch.nn as nn
import torch.optim as optim
import itertools

from generator import Generator
from discriminator import Discriminator
from image_pool import ImagePool


class CycleGAN(nn.Module):
    def __init__(self, opt=None, mode='train'):
        super(CycleGAN, self).__init__()
        if torch.cuda.is_available() and opt['cuda'] is not None:
            self.device = torch.device(opt['cuda'])
        else:
            self.device = torch.device('cpu')
        self.device = torch.device(opt['cuda'])
        self.lambda_identity = opt['lambda_identity']
        self.lambda_A = opt['lambda_A']
        self.lambda_B = opt['lambda_B']
        self.netG_A = Generator(opt['input_nc'], opt['output_nc']).to(self.device)
        self.netG_B = Generator(opt['input_nc'], opt['output_nc']).to(self.device)
        if mode == 'train':
            self.netD_A = Discriminator(opt['output_nc']).to(self.device)
            self.netD_B = Discriminator(opt['output_nc']).to(self.device)
            self.fake_A_pool = ImagePool(opt['pool_size'])  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt['pool_size'])  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = GANLoss().to(self.device)  # define GAN loss.
            self.criterionCycle = nn.L1Loss()
            self.criterionIdt = nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = optim.Adam(
                itertools.chain(
                    self.netG_A.parameters(),
                    self.netG_B.parameters()
                ),
                lr=opt['learning_rate'],
                betas=(opt['beta1'], 0.999)
            )
            self.optimizer_D = optim.Adam(
                itertools.chain(
                    self.netD_A.parameters(),
                    self.netD_B.parameters()
                ),
                lr=opt['learning_rate'],
                betas=(opt['beta1'], 0.999)
            )
            self.optimizers = [self.optimizer_G, self.optimizer_D]

    def forward(self, real_A, real_B):
        fake_B = self.netG_A(real_A)  # G_A(A)
        rec_A = self.netG_B(fake_B)   # G_B(G_A(A))
        fake_A = self.netG_B(real_B)  # G_B(B)
        rec_B = self.netG_A(fake_A)   # G_A(G_B(B))

        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        loss_G = self.backward_G(real_A, real_B, fake_A, fake_B, rec_A, rec_B)
        self.optimizer_G.step()

        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        loss_D_A = self.backward_D_A(real_B, fake_B)
        loss_D_B = self.backward_D_B(real_A, fake_A)
        self.optimizer_D.step()

        total_loss = loss_G + loss_D_A + loss_D_B
        return loss_G.item(), loss_D_A.item(), loss_D_B.item(), total_loss.item()

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)

        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self, real_B, fake_B):
        fake_B = self.fake_B_pool.query(fake_B)
        loss_D_A = self.backward_D_basic(self.netD_A, real_B, fake_B)
        return loss_D_A

    def backward_D_B(self, real_A, fake_A):
        fake_A = self.fake_B_pool.query(fake_A)
        loss_D_B = self.backward_D_basic(self.netD_B, real_A, fake_A)
        return loss_D_B

    def backward_G(self, real_A, real_B, fake_A, fake_B, rec_A, rec_B):
        lambda_idt = self.lambda_identity
        lambda_A = self.lambda_A
        lambda_B = self.lambda_B

        # Identity loss
        idt_A = self.netG_A(real_B)
        loss_idt_A = self.criterionIdt(idt_A, real_B) * lambda_B * lambda_idt
        idt_B = self.netG_B(real_A)
        loss_idt_B = self.criterionIdt(idt_B, real_A) * lambda_A * lambda_idt

        # In generator, fake image is True label
        loss_G_A = self.criterionGAN(self.netD_A(fake_B), True)
        loss_G_B = self.criterionGAN(self.netD_B(fake_A), True)

        loss_cycle_A = self.criterionCycle(rec_A, real_A) * lambda_A
        loss_cycle_B = self.criterionCycle(rec_B, real_B) * lambda_B
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        loss_G.backward()
        return loss_G

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss


if __name__ == '__main__':
    import config
    opt = config.opt
    model = CycleGAN(opt)
    device = torch.device(opt['cuda']) if torch.cuda.is_available() and opt['cuda'] is not None else torch.device('cpu')
    A = torch.rand(4, 3, 256, 256).to(device)
    B = torch.rand(4, 3, 256, 256).to(device)
    print(model(A,B))
