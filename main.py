import argparse

from networks import Generator, Discriminator
from utils import ReplayBuffer, LambdaLR, Logger, weights_init_normal
from datasets import ImageDataset

def main(opt):
    # select device
    device = torch.device('cuda') if torch.cuda.is_availabel() else torch.device('cpu')

    # create & init model
    netG_A2B = Generator(opt.input_nc, opt.output_nc).to(device)
    netG_B2A = Generator(opt.output_nc, opt.input_nc).to(device)
    netD_A = Discriminator(opt.input_nc).to(device)
    netD_B = Discriminator(opt.output_nc).to(device)
    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)
    models = [netG_A2B, netG_B2A, net_D_A, net_D_B]

    # Loss
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    criterions = [criterion_GAN, criterion_cycle, criterion_identity]

    # optimizer
    optimizer_G = optim.Adam(
        chain(netG_A2B.parameters(), netG_B2A.parameters()),
        lr=opt.lr,
        betas=(0.5, 0.999)
    )
    optimizer_D_A = optim.Adam(
        netD_A.parameters(),
        lr=opt.lr,
        betas=(0.5, 0.999)
    )
    optimizer_D_B = optim.Adam(
        netD_A.parameters(),
        lr=opt.lr,
        betas=(0.5, 0.999)
    )
    optimizers = [optimizer_G, optimizer_D_A, optimizer_D_B]

    # scheduler
    lr_scheduler_G = optim.lr_scheduler.LambdaLR(
        optimizer_G,
        lr_lambda=LambdaLR(opt.n_epochs,
                           opt.epoch,
                           opt.decay_epoch).step
    )
    lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(
        optimizer_D_A,
        lr_lambda=LambdaLR(opt.n_epochs,
                           opt.epoch,
                           opt.decay_epoch).step
    )
    lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(
        optimizer_D_B,
        lr_lambda=LambdaLR(opt.n_epochs,
                           opt.epoch,
                           opt.decay_epoch).step
    )
    schedulers = [lr_scheduler_G, lr_scherduler_D_A, lr_scheduler_D_B]

    # Inputs & Target memory allocation
    Tensor = torch.cuda.FlotTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
    target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # dataloader
    dataloader = DataLoader(
        ImageDataset(opt.dataroot, unaligned=True),
        batch_size=opt.batchsize,
        shuffle=True,
        num_workers=opt.n_workers
    )

    # logger
    logger = Logger(opt.n_epochs, len(dataloader))

    if opt.mode == 'train':
        train(model, optimizer
