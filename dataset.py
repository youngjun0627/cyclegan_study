from torch.utils.data import Dataset
import cv2
import albumentations
import glob
import os
import random


def create_transforms(size, crop_size, mode):
    if mode == 'train':
        transformlist = [
            albumentations.Resize(size, size, interpolation=3),
            albumentations.RandomCrop(crop_size, crop_size)
        ]
        transformlist += [albumentations.OneOf([
            albumentations.HorizontalFlip(),
            albumentations.VerticalFlip(),
            albumentations.Rotate(limit=90)
            ])
        ]
        transformlist += [albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]

    elif mode == 'val' or mode == 'test':
        transformlist = [
            albumentations.Resize(size, size, interpolation=3),
            albumentations.CenterCrop(crop_size, crop_size),
            albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]

    return albumentations.Compose(transformlist)


class ImageDataset(Dataset):
    def __init__(self, root, size, crop_size, unaligned=False, mode='train'):  # mode is one of [train, val, test]
        self.transform = create_transforms(size, crop_size, mode)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, mode, 'A') + '/*.jpg'))
        self.files_B = sorted(glob.glob(os.path.join(root, mode, 'B') + '/*.jpg'))

    def __getitem__(self, index):
        image_A = cv2.imread(self.files_A[index % len(self.files_A)], cv2.IMREAD_COLOR)
        if self.unaligned:
            image_B = cv2.imread(self.files_B[random.randint(0, len(self.files_B) - 1)], cv2.IMREAD_COLOR)
        else:
            image_B = cv2.imread(self.files_B[index % len(self.files_B)], cv2.IMREAD_COLOR)
        image_A = cv2.cvtColor(image_A, cv2.COLOR_BGR2RGB)
        image_B = cv2.cvtColor(image_B, cv2.COLOR_BGR2RGB)

        image_A = self.transform(image=image_A)['image']
        image_B = self.transform(image=image_B)['image']

        image_A = self.to_tensor(image_A)
        image_B = self.to_tensor(image_B)
        return image_A, image_B

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def to_tensor(self, x):
        return x.transpose((2, 0, 1))


if __name__ == '__main__':
    import config
    opt = config.opt
    data = ImageDataset(
            opt['dataroot'],
            opt['size'],
            opt['crop_size'],
            unaligned=True if opt['dataset_mode'] is 'unaligned' else False,
            mode='test'
    )
    print(data[0][0].shape, data[0][1].shape)
