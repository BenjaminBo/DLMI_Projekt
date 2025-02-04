import os
import shutil
from sklearn.model_selection import train_test_split

input_folders = {'healthy': 'data/healthy', 'bleeding': 'data/bleeding'}
output_dir = 'data'  # Destination folder

# Split ratio
test_size = 0.2  # 20% for testing

# Create train/test directories
for split in ['train', 'test']:
    for category in input_folders.keys():
        os.makedirs(os.path.join(output_dir, split, category), exist_ok=True)

# Perform train/test split
for category, folder in input_folders.items():
    images = [img for img in os.listdir(folder) if os.path.isfile(os.path.join(folder, img))]
    train_imgs, test_imgs = train_test_split(images, test_size=test_size, random_state=42)
    
    # Move images to train and test folders
    for img in train_imgs:
        shutil.copy(os.path.join(folder, img), os.path.join(output_dir, 'train', category, img))
    for img in test_imgs:
        shutil.copy(os.path.join(folder, img), os.path.join(output_dir, 'test', category, img))

print("Train/Test split completed successfully!")