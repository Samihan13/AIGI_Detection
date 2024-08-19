# data_loader.py
from torchvision import datasets, transforms
import os

def get_transforms():
    """ Returns transformations for the datasets. """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }
    return data_transforms

def load_data(data_dir):
    """ Loads the CIFAKE datasets and returns DataLoaders for train and test sets. """
    transforms = get_transforms()
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), transforms[x])
        for x in ['train', 'test']
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True if x == 'train' else False, num_workers=4)
        for x in ['train', 'test']
    }
    return dataloaders
