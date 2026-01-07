import argparse
import json
import multiprocessing as mp
import os

import numpy as np
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from tianshou.data import Batch
from huggingface_hub import hf_hub_download
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.idm import IDM
from gr00t.utils.video import get_all_frames_and_timestamps
from gr00t.data.embodiment_tags import EmbodimentTag



def load_dataset_and_config(checkpoint_path, validation_dataset_path, video_indices):
    # Check if checkpoint_path is a HuggingFace model repo
    is_hf_repo = not os.path.exists(checkpoint_path) and '/' in checkpoint_path
    
    if is_hf_repo:
        # For HuggingFace repos, we need to download config differently
        try:
            # Download the config file from the repo
            config_file = hf_hub_download(
                repo_id=checkpoint_path,
                filename="experiment_cfg/conf.yaml",
                repo_type="model"
            )
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config from HuggingFace repo: {e}")
            raise
    else:
        # Original local path handling
        config_path = os.path.join(checkpoint_path, "experiment_cfg", "conf.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    
    cfg = OmegaConf.create(config)

    dataset_name = os.path.basename(validation_dataset_path).split(".")[0]
    embodiment = dataset_name
    print(f"Dataset name: {dataset_name}")

    modality_configs = cfg["modality_configs"][embodiment]
    if video_indices is not None:
        video_delta_indices = video_indices.split(" ")
        print(f"Using provided video_delta_indices: {video_delta_indices}")
        modality_configs["video"]["delta_indices"] = video_delta_indices
    modality_configs = instantiate(modality_configs)


    if "all_transforms" in cfg:
        transform = cfg["all_transforms"][embodiment]
    else:
        transform = cfg["train_dataset"]["all_transforms"][embodiment]
    
    # Filter out VideoColorJitter transform for inference
    if "transforms" in transform:
        filtered_transforms = []
        for t in transform["transforms"]:
            if t.get("_target_") != "groot.data.transform.VideoColorJitter":
                filtered_transforms.append(t)
            
        transform["transforms"] = filtered_transforms
    transform_inst = instantiate(transform)

    metadata_versions = cfg["metadata_versions"]
    metadata_version = metadata_versions[embodiment]

    if "gr1" in embodiment:
        embodiment_tag = EmbodimentTag.GR1_unified
    elif "franka" in embodiment:
        embodiment_tag = EmbodimentTag.FRANKA
    elif "so101" in embodiment:
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    elif "so100" in embodiment:
        embodiment_tag = EmbodimentTag.SO100
    elif "robocasa" in embodiment:
        embodiment_tag = EmbodimentTag.ROBOCASA
    elif "piper" in embodiment:
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    elif "piper double" in embodiment:
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        raise ValueError(f"Unknown embodiment: {embodiment}")

    dataset = LeRobotSingleDataset(
        dataset_path=validation_dataset_path,
        modality_configs=modality_configs,
        # metadata_version=metadata_version,
        transforms=transform_inst,
        embodiment_tag=embodiment_tag,
    )

    return cfg, dataset, modality_configs


def collate_fn(features_list, device):
    batch_dict = {}
    keys = features_list[0].keys()
    for key in keys:
        if key in ["images", "view_ids"]:
            vals = [f[key] for f in features_list]
            batch_dict[key] = torch.as_tensor(np.concatenate(vals), device=device)
        else:
            vals = [f[key] for f in features_list]
            batch_dict[key] = torch.as_tensor(np.stack(vals), device=device)

    return batch_dict

def get_step_data(dataset, trajectory_id, base_index):
    data = {}
    dataset.curr_traj_data = dataset.get_trajectory_data(trajectory_id)
    # Get the data for all modalities
    for modality in dataset.modality_keys:
        # Get the data corresponding to each key in the modality
        for key in dataset.modality_keys[modality]:
            if modality == "video":
                pass
            elif modality == "state" or modality == "action":
                data[key] = dataset.get_state_or_action(trajectory_id, modality, key, base_index)
            elif modality == "language":
                data[key] = dataset.get_language(trajectory_id, key, base_index)

    return data

def save_trajectory_data(trajectory_data, dataset, trajectory_id, output_dir):
    chunk_index = dataset.get_episode_chunk(trajectory_id)
    chunk_dir = f"chunk-{chunk_index:03d}"
    os.makedirs(os.path.join(output_dir, "data", chunk_dir), exist_ok=True)

    episode_id = f"episode_{int(trajectory_id):06d}"
    output_file_path = os.path.join(output_dir, "data", chunk_dir, f"{episode_id}.parquet")
    trajectory_data.to_parquet(output_file_path)

##
# Worker function, now handles either validation or writing based on a boolean flag.
##
def worker_func(
    gpu_id: int,
    traj_id_list: list,
    checkpoint_path: str,
    validation_dataset_path: str,
    output_dir: str,
    batch_size: int,
    num_workers: int = 1,
    video_indices=None,
    dataset=None,
    modality_configs=None,
):
    """This function runs in a separate process on GPU `gpu_id` and processes all `traj_id_list`."""
    # Load model on this GPU
    # from_pretrained should handle both local paths and HF model IDs
    print(f"Loading model from: {checkpoint_path}")
    model = IDM.from_pretrained(checkpoint_path)

    model.requires_grad_(False)
    model.eval()
    device = torch.device(f"cuda:{gpu_id}")
    model.to(device)
    

    for tid in tqdm(traj_id_list, desc=f"GPU {gpu_id}", position=gpu_id):
        action_dict = {}
        traj_data = dataset.get_trajectory_data(tid)
        length = len(traj_data)


        all_features = []

        # Create a simple prefetching mechanism with a thread pool
        from concurrent.futures import ThreadPoolExecutor

        def load_and_transform_step(step_idx, video_data):
            import torch
            from einops import rearrange

            step_data = get_step_data(dataset, tid, step_idx)
            timestamp = dataset.curr_traj_data["timestamp"].to_numpy()
            for key in video_data: 
                frames, whole_indices = video_data[key]
                step_indices = dataset.delta_indices[key] + step_idx
                step_indices = np.maximum(step_indices, 0)
                step_indices = np.minimum(step_indices, dataset.trajectory_lengths[tid] - 1)
                indices = np.array([np.where(np.isclose(whole_indices, val))[0][0] for val in timestamp[step_indices]])
                step_data[key] = frames[indices]

            output = dataset.transforms(step_data)
            return output
        

        video_data = {}
        for key in dataset.modality_keys["video"]: 
            video_path = dataset.get_video_path(tid, key.replace("video.", ""))
            video_backend = dataset.video_backend
            video_backend_kwargs = dataset.video_backend_kwargs
            frames, whole_indices = get_all_frames_and_timestamps(video_path.as_posix(), video_backend, video_backend_kwargs)
            video_data[key] = (frames, whole_indices)


        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for step_idx in range(length):
                future = executor.submit(load_and_transform_step, step_idx, video_data)
                futures.append(future)
            
            all_features = []
            for future in as_completed(futures):
                all_features.append(future.result())


        for start_idx in range(0, length, batch_size):
            end_idx = min(start_idx + batch_size, length)
            step_ids = list(range(start_idx, end_idx))

            batch_features = all_features[start_idx:end_idx]
            batch_dict = collate_fn(batch_features, device)

            with torch.no_grad():
                out = model.get_action(batch_dict)

            pred_actions = out["action_pred"].cpu()
            pred_actions = dataset.transforms.unapply(Batch(action=pred_actions))

            # Load modality.json to get the proper structure
            modality_json_path = os.path.join(validation_dataset_path, 'meta', 'modality.json')

            with open(modality_json_path, 'r') as f:
                modality_config = json.load(f)
            
            # Get action part configurations
            action_parts = modality_config.get('action', {})
            
            # Calculate total action dimension
            total_dim = 0
            for part, indices in action_parts.items():
                total_dim = max(total_dim, indices.get('end', 0))
            
            # Check if we have an action horizon dimension
            has_horizon = False
            action_horizon = 1
            sample_key = list(pred_actions.keys())[0]
            if len(pred_actions[sample_key].shape) == 3:
                has_horizon = True
                batch_size, action_horizon, _ = pred_actions[sample_key].shape
            else:
                batch_size = pred_actions[sample_key].shape[0]
            
            if has_horizon:
                final_actions = np.zeros((batch_size, action_horizon, total_dim))
                
                # Fill in the actions for parts that exist in pred_actions
                for part, indices in action_parts.items():
                    action_key = f'action.{part}'
                    start_idx = indices.get('start', 0)
                    end_idx = indices.get('end', 0)
                    
                    if action_key in pred_actions:
                        final_actions[:, :, start_idx:end_idx] = pred_actions[action_key]
            else:
                final_actions = np.zeros((batch_size, total_dim))
                
                # Fill in the actions for parts that exist in pred_actions
                for part, indices in action_parts.items():
                    action_key = f'action.{part}'
                    start_idx = indices.get('start', 0)
                    end_idx = indices.get('end', 0)
                    
                    if action_key in pred_actions:
                        final_actions[:, start_idx:end_idx] = pred_actions[action_key]
            
            pred_actions = final_actions

            # Not validating => we do the usual writing
            for i, s in enumerate(step_ids):
                for j in range(action_horizon):
                    if s+j >= length:
                        break
                    if s+j not in action_dict:
                        action_dict[s+j] = []
                    if has_horizon:
                        action_dict[s+j].append(pred_actions[i, j].flatten())
                    else:
                        action_dict[s+j].append(pred_actions[i].flatten())
        
        for s in action_dict:
            mean_action = np.mean(action_dict[s], axis=0)
            traj_data.at[s, "action"] = mean_action

        save_trajectory_data(traj_data, dataset, tid, output_dir)
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def validate_checkpoint(
    checkpoint_path,
    validation_dataset_path,
    output_dir=None,
    num_gpus=8,
    batch_size=16,
    max_episodes=None,
    num_workers=1,
    video_indices=None,
):
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} GPUs available.")
    if device_count < num_gpus:
        print(
            f"WARNING: You requested num_gpus={num_gpus} but only {device_count} GPUs are visible."
        )
        num_gpus = device_count

    # If not validation, create the output directory for saving predicted actions
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Will save predicted actions to: {output_dir}")

    _, dataset, modality_configs = load_dataset_and_config(checkpoint_path, validation_dataset_path, video_indices)

    dataset.transforms.eval()

    trajectory_ids = dataset.trajectory_ids
    if max_episodes is not None and max_episodes > 0:
        trajectory_ids = trajectory_ids[:max_episodes]

    print(f"Processing {len(trajectory_ids)} trajectories total.")

    # Split trajectories among GPUs
    chunk_size = (len(trajectory_ids) + num_gpus - 1) // num_gpus

    tasks_by_gpu = {}
    for i in range(num_gpus):
        start = i * chunk_size
        end = min(start + chunk_size, len(trajectory_ids))
        if start >= end:
            break
        gpu_traj_list = trajectory_ids[start:end]
        tasks_by_gpu[i] = gpu_traj_list

    try:
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for gpu_id, gpu_traj_list in tasks_by_gpu.items():
                future = executor.submit(
                    worker_func,
                    gpu_id,
                    gpu_traj_list,
                    checkpoint_path,
                    validation_dataset_path,
                    output_dir,
                    batch_size,
                    num_workers,
                    video_indices,
                    dataset,
                    modality_configs,
                )
                futures.append(future)
            
            # Wait for all futures to complete or handle interruption
            for future in as_completed(futures):
                try:
                    # Get the result to catch any exceptions
                    future.result()
                except Exception as e:
                    print(f"Error in worker process: {e}")
                    import traceback
                    traceback.print_exc()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Cleaning up...")
        # The context manager will handle cancellation of pending futures
    except Exception as e:
        print(f"Error in main process: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Force cleanup of CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # If not validating, optionally copy the meta & videos folders
    if output_dir:
        # Copy the metadata directory
        meta_src = os.path.join(validation_dataset_path, "meta")
        meta_dst = os.path.join(output_dir, "meta")
        if os.path.exists(meta_src):
            import shutil

            if not os.path.exists(meta_dst):
                shutil.copytree(meta_src, meta_dst)
            print(f"Copied metadata to: {meta_dst}")
        
            tasks_path = os.path.join(meta_dst, "tasks.jsonl")
            if os.path.exists(tasks_path):
                # Read the existing tasks
                tasks = []
                with open(tasks_path, "r") as f:
                    for line in f:
                        tasks.append(json.loads(line))

                # Update tasks with <DREAM> prefix if not already present
                updated_tasks = []
                for task in tasks:
                    if "task" in task and not task["task"].startswith("<DREAM>"):
                        task["task"] = f"<DREAM>{task['task']}"
                    updated_tasks.append(task)

                # Write the updated tasks back to the file
                with open(tasks_path, "w") as f:
                    for task in updated_tasks:
                        f.write(json.dumps(task) + "\n")

                print("Updated tasks.jsonl with <DREAM> prefix")


        # Copy the videos directory if it exists
        videos_src = os.path.join(validation_dataset_path, "videos")
        videos_dst = os.path.join(output_dir, "videos")
        if os.path.exists(videos_src):
            import shutil

            if not os.path.exists(videos_dst):
                shutil.copytree(videos_src, videos_dst)
            print(f"Copied videos to: {videos_dst}")

    return {"message": "Dataset with predicted actions written to disk"}


#
# Main
#
if __name__ == "__main__":
    # If you're doing spawn/forkserver across multiple processes:
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Validate a checkpoint on a dataset with multi-GPU parallelism"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the validation dataset")
    parser.add_argument(
        "--output_dir", type=str, help="Path to save the output dataset with predicted actions"
    )
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use in parallel")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference steps")
    parser.add_argument(
        "--max_episodes", type=int, default=None, help="Maximum number of episodes to validate"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of worker threads for data loading per GPU",
    )
    parser.add_argument(
        "--video_indices",
        type=str,
        default=None,
        help="Video frame indices to use for inference",
    )

    args = parser.parse_args()

    result = validate_checkpoint(
        checkpoint_path=args.checkpoint,
        validation_dataset_path=args.dataset,
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        max_episodes=args.max_episodes,
        num_workers=args.num_workers,
        video_indices=args.video_indices,
    )
