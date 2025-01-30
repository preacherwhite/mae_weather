import os
import torch
import tifffile
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
class MultiChannelDataset(Dataset):
    def __init__(self, data_path, transform=None, target_size=(288, 288)):
        self.data_path = data_path
        self.transform = transform
        self.target_size = target_size
        self.files = [f for f in os.listdir(data_path) if f.endswith('.tif')]
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.files[idx])
        # Load tif file
        img = tifffile.imread(file_path)
        
        assert len(img.shape) == 3, "Image shape should be 3"
        img = img.transpose(2, 0, 1)  # HWC to CHW
        #if channel is less than 22, add placeholder 0 channels
        if img.shape[0] < 22:
            img = np.pad(img, ((0, 22 - img.shape[0]), (0, 0), (0, 0)), mode='constant')
        # Pad or crop to target size
        c, h, w = img.shape
        th, tw = self.target_size
        
        if h < th:
            pad_h = th - h
            img = np.pad(img, ((0,0), (0,pad_h), (0,0)), mode='constant')
        if w < tw:
            pad_w = tw - w
            img = np.pad(img, ((0,0), (0,0), (0,pad_w)), mode='constant')
            
        if h > th or w > tw:
            start_h = (h - th) // 2
            start_w = (w - tw) // 2
            img = img[:, start_h:start_h+th, start_w:start_w+tw]

        if self.transform:
            # img = np.transpose(img, (1, 2, 0))  # CHW to HWC for PIL
            # img = (img * 255).astype(np.uint8)  # Scale to 0-255 range
            # img = Image.fromarray(img)

            img = torch.from_numpy(img).half()
            img = self.transform(img)
        
        #img = torch.from_numpy(img)
        return img, torch.zeros(1)  # Return dummy label if needed

def build_dataset(is_train, args):
    transform = None  # Define transforms if needed
    dataset = MultiChannelDataset(
        os.path.join(args.data_path, 'train' if is_train else 'val'),
        transform=transform,
        target_size=(args.input_size, args.input_size)
    )
    return dataset