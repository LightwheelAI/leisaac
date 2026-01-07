import os
import shutil
import re
import argparse

def process_mp4_files(source_dir, output_dir, recursive=False, task_name=None):
    """
    Process MP4 files from source_dir and its subdirectories, extract instructions from filenames,
    and save them to output_dir with appropriate structure.
    
    Args:
        source_dir: Directory containing MP4 files or subdirectories with MP4 files
        output_dir: Directory to save processed data
        recursive: If True, maintain the directory structure from source_dir in output_dir
        task_name: Optional task name to use as instruction when filename is just a number
    """
    if not recursive:
        # Original behavior: process all MP4 files into a single output directory
        labels_dir = os.path.join(output_dir, "labels")
        videos_dir = os.path.join(output_dir, "videos")
        
        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(videos_dir, exist_ok=True)
        
        # Find all MP4 files in source_dir
        mp4_files = []
        for file in os.listdir(source_dir):
            if file.endswith('.mp4'):
                mp4_files.append(os.path.join(source_dir, file))
        
        # Sort files numerically to ensure consistent ordering (1.mp4, 2.mp4, ..., 10.mp4, not 1.mp4, 10.mp4, 2.mp4)
        def sort_key(filepath):
            filename = os.path.basename(filepath)
            stem = os.path.splitext(filename)[0]
            try:
                return int(stem)
            except ValueError:
                return float('inf')
        mp4_files.sort(key=sort_key)

        # Process each MP4 file
        for idx, mp4_path in enumerate(mp4_files, 1):
            mp4_file = os.path.basename(mp4_path)
            
            # Extract instruction from filename (remove number prefix and .mp4 extension)
            instruction = re.sub(r'^\d+_', '', mp4_file).replace('.mp4', '').replace('_', ' ')
            
            # If instruction is just a number and task_name is provided, use task_name instead
            if task_name and instruction.isdigit():
                # Convert task_name from directory format (e.g., "Lift_the_red_cube_up") to readable format
                instruction = task_name.replace('_', ' ')
            
            label_file = os.path.join(labels_dir, f"{idx}.txt")
            with open(label_file, 'w') as f:
                f.write(instruction)
            
            # Copy video with new name
            target_video = os.path.join(videos_dir, f"{idx}.mp4")
            shutil.copy2(mp4_path, target_video)
            
            print(f"Processed {mp4_path} -> {idx}.mp4, instruction: {instruction}")
        
        print(f"Processed {len(mp4_files)} files. Results saved to {output_dir}")
    else:
        # Recursive behavior: maintain directory structure
        print(f"Processing in recursive mode, maintaining directory structure...")
        
        # Get all subdirectories in the source directory
        subdirs = []
        for item in os.listdir(source_dir):
            item_path = os.path.join(source_dir, item)
            if os.path.isdir(item_path):
                subdirs.append(item)
        
        if not subdirs:
            # If no subdirectories, process the source directory directly
            print(f"No subdirectories found in {source_dir}, processing directly...")
            process_mp4_files(source_dir, output_dir, recursive=False)
            return
        
        # Process each subdirectory
        for subdir in subdirs:
            source_subdir = os.path.join(source_dir, subdir)
            output_subdir = os.path.join(output_dir, subdir)
            
            # Create output subdirectory
            os.makedirs(output_subdir, exist_ok=True)
            
            # Process files in this subdirectory, passing the directory name as task_name
            process_mp4_files(source_subdir, output_subdir, recursive=False, task_name=subdir)
        
        print(f"Recursive processing complete. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MP4 files from source directory and save to output directory")
    parser.add_argument("--source_dir", type=str, required=True, help="Path to source directory containing MP4 files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--recursive", action="store_true", help="Process subdirectories recursively, maintaining directory structure")
    args = parser.parse_args()
    
    process_mp4_files(args.source_dir, args.output_dir, args.recursive)