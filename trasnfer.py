#%%
import os
import shutil

def transfer_mat_files(source_dir, target_dir):
    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Walk through the source directory
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.mat'):
                # Get the relative path from the source directory
                rel_path = os.path.relpath(root, source_dir)
                
                # Create the new path in the target directory
                new_dir = os.path.join(target_dir, rel_path)
                os.makedirs(new_dir, exist_ok=True)
                
                # Copy the file to the new location
                src_file = os.path.join(root, file)
                dst_file = os.path.join(new_dir, file)
                shutil.copy2(src_file, dst_file)
                print(f"Copied: {src_file} -> {dst_file}")

# Set the source and target directories
source_directory = '.'  # Current directory
target_directory = './footprints'

# Call the function to transfer .mat files
transfer_mat_files(source_directory, target_directory)

print("Transfer complete!")
