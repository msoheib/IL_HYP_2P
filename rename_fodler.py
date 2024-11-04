#%%
import os
import shutil

def copy_png_files_with_parent_folder_name():
    source_dir = '.'  # Current directory
    target_dir = 'figure_new'
    os.makedirs(target_dir, exist_ok=True)

    for root, _, files in os.walk(source_dir):
        if 'figure_new' in root:
            continue  # Skip the figure_new folder
        
        parent_folder = os.path.basename(root)
        for file in files:
            if file.lower().endswith('.png'):
                source_path = os.path.join(root, file)
                new_filename = f"{os.path.splitext(file)[0]}_{parent_folder}.png"
                target_path = os.path.join(target_dir, new_filename)
                shutil.copy2(source_path, target_path)

    print("PNG files copied successfully.")

# Run the function
copy_png_files_with_parent_folder_name()