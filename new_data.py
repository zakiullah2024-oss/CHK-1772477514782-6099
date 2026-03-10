import os
import shutil
import random

# Settings
source_dir = 'dataset'        
dest_dir = 'dataset_10k'      
images_per_class = 263        

print("⏳ 10,000 images cha dataset tayar hot ahe... thoda vel thaamba.")


os.makedirs(dest_dir, exist_ok=True)
total_copied = 0

for folder_name in os.listdir(source_dir):
    source_folder = os.path.join(source_dir, folder_name)
    
    if os.path.isdir(source_folder):
        dest_folder = os.path.join(dest_dir, folder_name)
        os.makedirs(dest_folder, exist_ok=True)
        
        all_images = os.listdir(source_folder)
        
        
        available_images = len(all_images)
        images_to_copy = min(images_per_class, available_images)
        
        
        selected_images = random.sample(all_images, images_to_copy)
        
        for img in selected_images:
            src_path = os.path.join(source_folder, img)
            dst_path = os.path.join(dest_folder, img)
            shutil.copy(src_path, dst_path)
            total_copied += 1
            
        print(f"📁 '{folder_name}' madhun {images_to_copy} images copy zale.")

print(f"\n✅ Yashasvi! Ekun {total_copied} images cha navin dataset '{dest_dir}' folder madhe tayar zala ahe.")
