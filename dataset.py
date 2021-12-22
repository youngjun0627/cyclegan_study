import torchvision.transforms as transforms
import torch.utils.data import Dataset

def create_transforms(mode):
    if mode == 'train':
        transforms = [
            transforms.Resize(int(opt.size*1.12), Image.BICUBIC), 
            transforms.RandomCrop(opt.size), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ]
    elif mode == 'test':
        transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ]
    return transforms

class ImageDataset(Dataset):
    def __init__(self, root, unaligned=False, mode='train'):
        self.transform = transforms.Compose(create_transforms(mode))
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        image_A = cv2.imread(self.files_A[index % len(self.files_A)], cv2.IMREAD_COLOR)
        if self.unaligned:
            image_B = cv2.imread(self.files_B[random.randint(0, len(self.files_B))], cv2.IMREAD_COLOR)
        else:
            image_B = cv2.imread(self.files_B[index % len(self.files_B)], cv2.IMREAD_COLOR)
        image_A = cv2.cvtColor(image_A, cv2.COLOR_BGR2RGB)
        image_B = cv2.cvtColor(image_B, cv2.COLOR_BGR2RGB)

        image_A = self.transform(image_A)
        image_B = self.transform(image_B)
        return image_A, image_B
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

