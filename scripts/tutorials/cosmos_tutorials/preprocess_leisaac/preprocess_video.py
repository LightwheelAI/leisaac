import os
import cv2
import numpy as np
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import shutil
import argparse
import concurrent.futures
from functools import partial
import decord
import imageio


def custom_crop_pad_resize_gr1(img, target_size=(256, 256)):
    """
    Custom crop, pad, and resize operation that maintains the aspect ratio.
    
    Args:
        img: Input frame
        original_width: Original video width
        original_height: Original video height
        target_size: Target resolution (width, height)
        
    Returns:
        Processed frame at target_size resolution
    """
    # For 832x480 videos, adjust the crop parameters proportionally from the 1280x800 example
    # Original crop for 1280x800: (310, 770, 110, 1130) - (top, bottom, left, right)
    original_height, original_width = img.shape[:2]


    # Calculate proportional crop values
    crop_top_ratio = 310 / 800
    crop_bottom_ratio = 770 / 800
    crop_left_ratio = 110 / 1280
    crop_right_ratio = 1130 / 1280
    
    # Apply ratios to the current dimensions
    crop_top = int(original_height * crop_top_ratio)
    crop_bottom = int(original_height * crop_bottom_ratio)
    crop_left = int(original_width * crop_left_ratio)
    crop_right = int(original_width * crop_right_ratio)
    
    # Ensure crop boundaries are within image dimensions
    crop_top = max(0, min(crop_top, original_height - 1))
    crop_bottom = max(crop_top + 1, min(crop_bottom, original_height))
    crop_left = max(0, min(crop_left, original_width - 1))
    crop_right = max(crop_left + 1, min(crop_right, original_width))
    
    # Crop the image
    img_cropped = img[crop_top:crop_bottom, crop_left:crop_right]
    
    # Calculate intermediate size while maintaining aspect ratio
    cropped_height, cropped_width = img_cropped.shape[:2]
    aspect_ratio = cropped_width / cropped_height
    
    # Resize to intermediate size (similar to 720x480 in the example)
    intermediate_height = 480
    intermediate_width = 720
    img_resized = cv2.resize(img_cropped, (intermediate_width, intermediate_height), cv2.INTER_AREA)
    
    # Pad to make it square
    if intermediate_width > intermediate_height:
        # Width is larger, pad height
        height_pad = (intermediate_width - intermediate_height) // 2
        img_pad = np.pad(
            img_resized, ((height_pad, height_pad), (0, 0), (0, 0)), mode="constant", constant_values=0
        )
    else:
        # Height is larger, pad width
        width_pad = (intermediate_height - intermediate_width) // 2
        img_pad = np.pad(
            img_resized, ((0, 0), (width_pad, width_pad), (0, 0)), mode="constant", constant_values=0
        )
    
    # Final resize to target size
    final_img = cv2.resize(img_pad, target_size, cv2.INTER_AREA)
    
    return final_img

def resize_with_padding(img, ratio=1.0, target_size=(256, 256)):
    # Original aspect ratio is 1280:800 (or 16:10)
    target_ratio = ratio
    
    h, w = img.shape[:2]
    current_ratio = w / h
    if target_ratio >= 1:
        # Width is the limiting factor
        new_w = target_size[0]
        new_h = int(new_w / target_ratio)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Add padding to height
        pad_top = (target_size[1] - new_h) // 2
        pad_bottom = target_size[1] - new_h - pad_top
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        # Height is the limiting factor
        new_h = target_size[1]
        new_w = int(new_h * target_ratio)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Add padding to width
        pad_left = (target_size[0] - new_w) // 2
        pad_right = target_size[0] - new_w - pad_left
        padded = cv2.copyMakeBorder(resized, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
    return padded

def extract_subimages_franka(frame, original_width, original_height):
    h, w = frame.shape[:2]  # h=480, w=832
    
    # Calculate dimensions for even division
    half_width = w // 2  
    half_height = h // 2  
    
    # Extract subimages
    image_side_0 = frame[:half_height, :half_width]     # Top-left (240x416)
    image_side_1 = frame[:half_height, half_width:]     # Top-right (240x416)
    wrist_image = frame[half_height:, :half_width]      # Bottom-left (240x416)


    image_side_0 = cv2.resize(image_side_0, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    image_side_1 = cv2.resize(image_side_1, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    wrist_image = cv2.resize(wrist_image, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    
    return image_side_0, image_side_1, wrist_image


def extract_subimages(frame, ratio):
    """Extract subimages from a frame and resize to 256x256 while preserving aspect ratio with padding."""
    h, w = frame.shape[:2]  # h=480, w=832
    
    # Calculate dimensions for even division
    half_width = w // 2  # 416
    half_height = h // 2  # 240
    
    # Extract subimages
    image_side_0 = frame[:half_height, :half_width]     # Top-left (240x416)
    image_side_1 = frame[:half_height, half_width:]     # Top-right (240x416)
    wrist_image = frame[half_height:, :half_width]      # Bottom-left (240x416)

    image_side_0 = resize_with_padding(image_side_0, ratio)
    image_side_1 = resize_with_padding(image_side_1, ratio)
    wrist_image = resize_with_padding(wrist_image, ratio)
    
    return image_side_0, image_side_1, wrist_image

def process_batch_frames(frames, output_videos, src_path, dataset, original_width, original_height):
    """Process a batch of frames."""
    ratio = original_width / original_height
    for frame in frames:
        # Extract subimages
        if dataset == 'robocasa':
            image_side_0, image_side_1, wrist_image = extract_subimages(frame, ratio)
            output_videos['observation.images.left_view'].append_data(image_side_0)
            output_videos['observation.images.right_view'].append_data(image_side_1)
            output_videos['observation.images.wrist_view'].append_data(wrist_image)
        elif dataset == 'gr1':
            image = custom_crop_pad_resize_gr1(frame)
            output_videos['observation.images.ego_view'].append_data(image)
        elif dataset == 'franka':
            image_side_0, image_side_1, wrist_image = extract_subimages(frame, ratio)
            output_videos['observation.images.exterior_image_1_left_pad_res256_freq15'].append_data(image_side_0)
            output_videos['observation.images.exterior_image_2_left_pad_res256_freq15'].append_data(image_side_1)
            output_videos['observation.images.wrist_image_left_pad_res256_freq15'].append_data(wrist_image)
        elif dataset == 'so100':
            image = resize_with_padding(frame, ratio)
            output_videos['observation.images.front'].append_data(image)
        elif dataset == 'so101':
            image = resize_with_padding(frame, ratio)
            output_videos['observation.images.front'].append_data(image)
        elif dataset == 'piper':
            # Piper: 720x720 -> resize to 256x256 (square, no padding needed)
            image = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
            output_videos['observation.images.agentview_all'].append_data(image)
        elif dataset == 'piper double':
            image = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
            output_videos['observation.images.agentview_all'].append_data(image)
        else:
            raise ValueError(f"Unknown task: {src_path}")

def process_video(args):
    """Process a single video file."""
    src_path, dst_dir, video_name, dataset, original_width, original_height = args
    
    # Create output directories if they don't exist
    if dataset == 'robocasa':
        output_dirs = {
            'observation.images.left_view': os.path.join(dst_dir, 'videos', 'observation.images.left_view'),
            'observation.images.right_view': os.path.join(dst_dir, 'videos', 'observation.images.right_view'),
            'observation.images.wrist_view': os.path.join(dst_dir, 'videos', 'observation.images.wrist_view'),
        }
    elif dataset == 'gr1':
        output_dirs = {
            'observation.images.ego_view': os.path.join(dst_dir, 'videos', 'observation.images.ego_view')
        }
    elif dataset == 'franka':
        output_dirs = {
            'observation.images.exterior_image_1_left_pad_res256_freq15': os.path.join(dst_dir, 'videos', 'observation.images.exterior_image_1_left_pad_res256_freq15'),
            'observation.images.exterior_image_2_left_pad_res256_freq15': os.path.join(dst_dir, 'videos', 'observation.images.exterior_image_2_left_pad_res256_freq15'),
            'observation.images.wrist_image_left_pad_res256_freq15': os.path.join(dst_dir, 'videos', 'observation.images.wrist_image_left_pad_res256_freq15'),
        }
    elif dataset == 'so100':
        output_dirs = {
            'observation.images.front': os.path.join(dst_dir, 'videos', 'observation.images.front'),
        }
    elif dataset == 'so101':
        output_dirs = {
            'observation.images.front': os.path.join(dst_dir, 'videos', 'observation.images.front'),
        }
    elif dataset == 'piper':
        output_dirs = {
            'observation.images.agentview_all': os.path.join(dst_dir, 'videos', 'observation.images.agentview_all'),
        }
    elif dataset == 'piper double':
        output_dirs = {
            'observation.images.agentview_all': os.path.join(dst_dir, 'videos', 'observation.images.agentview_all'),
        }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Open the source video
    vr = decord.VideoReader(src_path)
    
    # Get video properties
    fps  = vr.get_avg_fps()
    frame_count = len(vr)
    
    # Create VideoWriter objects for each subimage
    output_videos = {}
    for name, dir_path in output_dirs.items():
        output_path = os.path.join(dir_path, f"{video_name}.mp4")
        output_videos[name] = imageio.get_writer(output_path, fps=fps)

    
    # Process frames in batches
    batch_size = 32  # Adjust based on available memory
    frames_batch = []
    
    # Process each frame with progress bar
    pbar = tqdm(total=frame_count, desc=f"Processing {video_name}", leave=False)
    
    for frame in vr:
        frames_batch.append(frame.asnumpy())
        
        # Process batch when it reaches the desired size
        if len(frames_batch) >= batch_size:
            process_batch_frames(frames_batch, output_videos, src_path, dataset, original_width, original_height)
            frames_batch = []
            pbar.update(batch_size)
    
    # Process remaining frames
    if frames_batch:
        process_batch_frames(frames_batch, output_videos, src_path, dataset, original_width, original_height)
        pbar.update(len(frames_batch))
    
    # Close progress bar
    pbar.close()
    
    # Release resources
    for writer in output_videos.values():
        writer.close()


def copy_labels(src_dir, dst_dir):
    """Copy label files from source to destination."""
    src_labels_dir = os.path.join(src_dir, 'labels')
    dst_labels_dir = os.path.join(dst_dir, 'labels')
    
    if os.path.exists(src_labels_dir):
        os.makedirs(dst_labels_dir, exist_ok=True)
        for label_file in os.listdir(src_labels_dir):
            if label_file.endswith('.txt'):
                shutil.copy2(
                    os.path.join(src_labels_dir, label_file),
                    os.path.join(dst_labels_dir, label_file)
                )

def process_subdirectory(subdir, src_dir, dst_dir, num_workers, max_videos=None, dataset=None, original_width=None, original_height=None):
    """Process a single subdirectory."""
    print(f"Processing subdirectory: {src_dir}, {subdir}")
    src_subdir = os.path.join(src_dir, subdir)
    dst_subdir = os.path.join(dst_dir, subdir)
    
    # Copy label files
    copy_labels(src_subdir, dst_subdir)
    
    # Process videos
    src_videos_dir = os.path.join(src_subdir, 'videos')
    if os.path.exists(src_videos_dir):
        video_files = [f for f in os.listdir(src_videos_dir) if f.endswith('.mp4')]
        
        # Sort files numerically to ensure consistent ordering (1.mp4, 2.mp4, ..., 10.mp4, not 1.mp4, 10.mp4, 2.mp4)
        def sort_key(filename):
            stem = os.path.splitext(filename)[0]
            try:
                return int(stem)
            except ValueError:
                return float('inf')
        video_files.sort(key=sort_key)
        
        # Limit number of videos if max_videos is specified
        if max_videos is not None:
            video_files = video_files[:max_videos]
            print(f"Processing limited set of {len(video_files)} videos in {subdir}")
        
        # Prepare arguments for multiprocessing
        args_list = [
            (os.path.join(src_videos_dir, video_file), dst_subdir, os.path.splitext(video_file)[0], dataset, original_width, original_height)
            for video_file in video_files
        ]
        
        # Process videos in parallel
        with mp.Pool(num_workers) as pool:
            list(tqdm(pool.imap(process_video, args_list), total=len(args_list), desc=f"Processing {subdir}"))

def process_directory(src_dir, dst_dir, num_workers=None, num_subdirs_parallel=1, max_videos=None, dataset=None, original_width=None, original_height=None, recursive=False):
    """Process all videos in the source directory."""
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() // 2)  # Leave some cores for system
    
    # Create destination directory structure
    os.makedirs(dst_dir, exist_ok=True)


    if recursive:
        # Get all subdirectories in the source directory
        subdirs = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    else:
        # Get all subdirectories in the source directory
        subdirs = ['']
    
    # Process subdirectories in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_subdirs_parallel) as executor:
        process_subdir_fn = partial(process_subdirectory, 
                                  src_dir=src_dir, 
                                  dst_dir=dst_dir, 
                                  num_workers=num_workers // num_subdirs_parallel,
                                  max_videos=max_videos,
                                  dataset=dataset,
                                  original_width=original_width,
                                  original_height=original_height)
        list(tqdm(executor.map(process_subdir_fn, subdirs), total=len(subdirs), desc="Processing subdirectories"))

def main():
    parser = argparse.ArgumentParser(description='Split videos into subimages and save them.')
    parser.add_argument('--src_dir', type=str, default='/mnt/amlfs-01/home/seonghyeony/data/dreams_robocasa_70K_0228',
                        help='Source directory containing videos')
    parser.add_argument('--dst_dir', type=str, default='/mnt/amlfs-01/home/seonghyeony/data/dreams_robocasa_70K_0228_split',
                        help='Destination directory for processed videos')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of worker processes (default: half of CPU cores)')
    parser.add_argument('--parallel_subdirs', type=int, default=1,
                        help='Number of subdirectories to process in parallel')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of frames to process in a batch')
    parser.add_argument('--max_videos', type=int, default=None,
                        help='Maximum number of videos to process per subdirectory (for debugging)')
    parser.add_argument('--dataset', type=str, default='robocasa',
                        help='Dataset name', choices=['robocasa', 'gr1', 'franka', 'so100', 'so101', 'piper', 'piper double'])
    parser.add_argument("--recursive", action="store_true", help="Process subdirectories recursively, maintaining directory structure")
    parser.add_argument("--original_width", type=int, default=1280, help="Original width of the video")
    parser.add_argument("--original_height", type=int, default=800, help="Original height of the video")
    
    args = parser.parse_args()
    # Set OpenCV thread optimization
    cv2.setNumThreads(1)  # Disable OpenCV's internal multithreading to avoid conflicts
    
    process_directory(args.src_dir, args.dst_dir, args.workers, args.parallel_subdirs, 
                     max_videos=args.max_videos, dataset=args.dataset, original_width=args.original_width, original_height=args.original_height, recursive=args.recursive)

if __name__ == "__main__":
    main()
