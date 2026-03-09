import os
import shutil
import random

# Settings
source_dir = 'dataset'        # Tujhya junya mothya dataset che nav
dest_dir = 'dataset_10k'      # Navin dataset che nav
images_per_class = 263        # 263 * 38 = javalpas 10,000 images

print("⏳ 10,000 images cha dataset tayar hot ahe... thoda vel thaamba.")

# Navin folder banavne
os.makedirs(dest_dir, exist_ok=True)
total_copied = 0

for folder_name in os.listdir(source_dir):
    source_folder = os.path.join(source_dir, folder_name)
    
    if os.path.isdir(source_folder):
        dest_folder = os.path.join(dest_dir, folder_name)
        os.makedirs(dest_folder, exist_ok=True)
        
        all_images = os.listdir(source_folder)
        
        # Jar ekhadya folder madhe 263 peksha kami images astil, tar aslele sagale gheu
        available_images = len(all_images)
        images_to_copy = min(images_per_class, available_images)
        
        # Randomly images select karne
        selected_images = random.sample(all_images, images_to_copy)
        
        for img in selected_images:
            src_path = os.path.join(source_folder, img)
            dst_path = os.path.join(dest_folder, img)
            shutil.copy(src_path, dst_path)
            total_copied += 1
            
        print(f"📁 '{folder_name}' madhun {images_to_copy} images copy zale.")

print(f"\n✅ Yashasvi! Ekun {total_copied} images cha navin dataset '{dest_dir}' folder madhe tayar zala ahe.")