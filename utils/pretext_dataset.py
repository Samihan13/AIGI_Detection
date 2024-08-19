# utils/pretext_dataset.py
import os
import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class RotationDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.images = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
        self.angles = [0, 90, 180, 270]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        angle = random.choice(self.angles)
        rotated_image = image.rotate(angle)
        
        if self.transform:
            rotated_image = self.transform(rotated_image)
        
        angle_label = self.angles.index(angle)
        return rotated_image, angle_label
