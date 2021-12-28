import config
from utils import LambdaLR, save_model
from dataset import ImageDataset
from gan import CycleGAN
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import argparse
import cv2
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', help='Select mode')
parser.add_argument('--image_path', type=str, default='b.jpg', help='When test mode, define image-path')

def train(opt):
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

def generate_image(opt, image_path):
    if torch.cuda.is_available() and opt['cuda'] is not None:
        device = torch.device(opt['cuda'])
    else:
        device = torch.device('cpu')
    # create & init model
    model = CycleGAN(opt)
    model.load_state_dict(torch.load('latest.pth'))
    model.eval()

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    img = torch.tensor(img, dtype=torch.float32).to(device)
    
    reimage_path = '/mnt/data/guest0/datasets/cezanne2photo/test/A/00010.jpg'
    re_img = cv2.imread(reimage_path, cv2.IMREAD_COLOR)
    re_img = cv2.cvtColor(re_img, cv2.COLOR_BGR2RGB)
    re_img = cv2.resize(re_img, (256, 256))
    re_img = np.transpose(re_img, (2, 0, 1))
    re_img = np.expand_dims(re_img, 0)
    re_img = torch.tensor(re_img, dtype=torch.float32).to(device)
    gen_images = model(img, re_img, mode='test')
    for idx, image in enumerate(gen_images):
        save_image(image, './{}.png'.format(idx))

if __name__ == '__main__':
    opt = config.opt
    arguments = parser.parse_args()
    if arguments.mode == 'train':
        train(opt)
    elif arguments.mode == 'test':
        image_path = arguments.image_path
        generate_image(opt, image_path)
