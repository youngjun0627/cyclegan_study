import config
from utils import LambdaLR, save_model
from dataset import ImageDataset
from gan import CycleGAN
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def main(opt):
    if torch.cuda.is_available() and opt['cuda'] is not None:
        device = torch.device(opt['cuda'])
    else:
        device = torch.device('cpu')
    # create & init model
    model = CycleGAN(opt)

    # scheduler
    lr_scheduler_G = optim.lr_scheduler.LambdaLR(
        model.optimizer_G,
        lr_lambda=LambdaLR(
            opt['n_epochs'],
            opt['offset'],
            opt['decay_epoch']
        ).step
    )
    lr_scheduler_D = optim.lr_scheduler.LambdaLR(
        model.optimizer_D,
        lr_lambda=LambdaLR(
            opt['n_epochs'],
            opt['offset'],
            opt['decay_epoch']
        ).step
    )
    schedulers = [lr_scheduler_G, lr_scheduler_D]

    # dataloader
    train_dataloader = DataLoader(
        ImageDataset(
            opt['dataroot'],
            opt['size'],
            opt['crop_size'],
            unaligned=True if opt['dataset_mode'] == 'unaligned' else False,
            mode='train'
        ),
        batch_size=opt['batch_size'],
        shuffle=True,
        num_workers=opt['n_workers']
    )
    validation_dataloader = DataLoader(
        ImageDataset(
            opt['dataroot'],
            opt['size'],
            opt['crop_size'],
            unaligned=True,
            mode='test'
        ),
        batch_size=opt['batch_size'],
        shuffle=False,
        num_workers=opt['n_workers']
    )

    prev_loss = 99999999
    for epoch in range(opt['n_epochs']):
        if epoch % opt['eval_num'] == 0:  # validate
            model.eval()
            for A, B in tqdm(validation_dataloader):
                A = A.to(device)
                B = B.to(device)
                loss_G, loss_D_A, loss_D_B, total_loss = model(A, B)
            if total_loss < prev_loss:
                prev_loss = total_loss
                save_model(model)
            model.train()
        else:
            for A, B in tqdm(train_dataloader):
                A = A.to(device)
                B = B.to(device)
                loss_G, loss_D_A, loss_D_B, total_loss = model(A, B)
        for scheduler in schedulers:
            scheduler.step()

        print('Epoch: {} \t losses: {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(
            epoch,
            loss_G,
            loss_D_A,
            loss_D_B, 
            total_loss
        ))

if __name__ == '__main__':
    opt = config.opt
    main(opt)
