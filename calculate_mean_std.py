import numpy as np
import tifffile
from tqdm import tqdm
import os

def calculate_dataset_stats(data_path):
    files = [f for f in os.listdir(data_path) if f.endswith('.tif')]
    
    # Counters for different channel counts
    total_files = 0
    valid_files = 0
    
    # First pass: calculate mean
    channel_sums = None
    pixel_count = 0
    
    print("Calculating means...")
    for file in tqdm(files):
        img = tifffile.imread(os.path.join(data_path, file)).astype(np.float32)
        if len(img.shape) == 2:
            img = img[None]
        elif len(img.shape) == 3:
            img = img.transpose(2, 0, 1)
        
        total_files += 1
        
        # Skip images with 19 channels
        if img.shape[0] != 22:
            os.remove(os.path.join(data_path, file))
            continue
            
        valid_files += 1
        
        if channel_sums is None:
            channel_sums = np.zeros(img.shape[0], dtype=np.float64)
            
        channel_sums += img.sum(axis=(1, 2))
        pixel_count += img.shape[1] * img.shape[2]
    
    if pixel_count == 0:
        raise ValueError("No valid 22-channel images found in the dataset")
        
    means = channel_sums / pixel_count
    
    # Second pass: calculate std
    channel_squared_diff_sums = np.zeros_like(channel_sums)
    
    print("Calculating standard deviations...")
    for file in tqdm(files):
        img = tifffile.imread(os.path.join(data_path, file)).astype(np.float32)
        if len(img.shape) == 2:
            img = img[None]
        elif len(img.shape) == 3:
            img = img.transpose(2, 0, 1)
            
        if img.shape[0] != 22:
            os.remove(os.path.join(data_path, file))
            continue
            
        for c in range(img.shape[0]):
            channel_squared_diff_sums[c] += ((img[c] - means[c]) ** 2).sum()
    
    stds = np.sqrt(channel_squared_diff_sums / pixel_count)
    
    # Calculate percentage of valid files
    valid_percentage = (valid_files / total_files) * 100 if total_files > 0 else 0
    
    return means, stds, valid_percentage

if __name__ == "__main__":
    data_path = "/media/staging2/dhwang/wildfire_data/wildfire_subset"  # Replace with your data path
    means, stds, valid_percentage = calculate_dataset_stats(data_path)
    print("\nChannel means:", means)
    print("Channel stds:", stds)
    print(f"\nPercentage of 22-channel images: {valid_percentage:.2f}%")
    
    # Save stats
    np.save('channel_means.npy', means)
    np.save('channel_stds.npy', stds)