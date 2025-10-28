"""
FOCUS Data Processing and I/O Module

This module handles data processing, video loading, similarity computation, and result output
for the FOCUS keyframe extraction algorithm.
"""

import os
import json
import argparse
import datetime
import random
import time
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch
import ray
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm

from lavis.models import load_model_and_preprocess
from focus import run_focus_algorithm


# ============================================================================
# Video Processing Functions
# ============================================================================

def blip_batch_similarity(frame_indices: List[int], vr: VideoReader, 
                         vis_processors, model, device: str, txt, batch_size: int) -> List[float]:
    """
    Compute BLIP similarity scores for a batch of frame indices.
    
    Args:
        frame_indices: List of frame indices to process
        vr: VideoReader object
        vis_processors: Vision processors for BLIP
        model: BLIP model
        device: Device to run inference on
        txt: Processed text input
        batch_size: Batch size for processing
        
    Returns:
        List of similarity scores
    """
    similarities = []
    for i in range(0, len(frame_indices), batch_size):
        batch_indices = frame_indices[i:i+batch_size]
        batch_images = []
        for idx in batch_indices:
            raw_image = vr[idx].numpy()
            raw_image = Image.fromarray(raw_image)
            img = vis_processors["eval"](raw_image).unsqueeze(0)
            batch_images.append(img)
        if batch_images:
            batch_tensor = torch.cat(batch_images, dim=0).to(device)
            with torch.no_grad():
                blip_output, _ = model({"image": batch_tensor, "text_input": txt}, match_head="itm-e")
                blip_scores = torch.nn.functional.softmax(blip_output, dim=1)
                batch_similarities = [blip_scores[j, 1].item() for j in range(len(batch_indices))]
                similarities.extend(batch_similarities)
    return similarities


def process_single_video(vr: VideoReader, text: str, vis_processors, text_processors, 
                       model, device: str, args, rng: np.random.Generator) -> Tuple[List[int], Dict]:
    """
    Process a single video with FOCUS algorithm.
    
    Args:
        vr: VideoReader object
        text: Query text
        vis_processors: Vision processors
        text_processors: Text processors
        model: BLIP model
        device: Device to run inference on
        args: Algorithm arguments
        rng: Random number generator
        
    Returns:
        Tuple of (selected_frames, sampling_details)
    """
    from focus import setup_arms, coarse_sampling_in_arms, fine_sampling_in_arms
    
    total_frames = len(vr)
    fps = float(vr.get_avg_fps())
    video_duration = float(total_frames) / max(1.0, fps)

    # Adaptive min-gap calculation
    avg_spacing_sec = video_duration / max(1, args.num_keyframes)
    if avg_spacing_sec <= float(args.disable_gap_below_sec):
        auto_min_gap_sec = 0.0
    else:
        gap_from_ratio = float(args.gap_ratio_of_avg) * avg_spacing_sec
        auto_min_gap_sec = min(gap_from_ratio, float(args.min_gap_sec))

    txt = text_processors["eval"](text)

    # Setup arms and perform coarse sampling
    coarse_stride = max(1, int(round(args.coarse_every_sec * fps)))
    desired_coarse = max(args.min_coarse_segments, int(np.ceil(video_duration / max(1e-6, args.coarse_every_sec))))
    if desired_coarse > 0:
        coarse_stride = max(1, int(np.floor(total_frames / desired_coarse)))

    arms = setup_arms(total_frames, coarse_stride)
    all_coarse_indices = coarse_sampling_in_arms(arms, args.extra_samples_per_region, rng)
    all_coarse_similarities = blip_batch_similarity(all_coarse_indices, vr, vis_processors, model, device, txt, args.batch_size)

    # Fine sampling
    fine_stride = max(1, int(round(args.fine_every_sec * fps)))
    fine_stride = min(fine_stride, coarse_stride)

    if args.region_half_window_sec is None:
        effective_coarse_sec = max(1e-6, video_duration / max(1, desired_coarse))
        region_half_window_sec = max(1.0 / fps, effective_coarse_sec / 2.0)
    else:
        region_half_window_sec = args.region_half_window_sec
    region_half_window = max(1, int(round(region_half_window_sec * fps)))

    # Run FOCUS algorithm
    selected_frames, arms, arm_selection_probs, arm_selection_counts = run_focus_algorithm(
        total_frames=total_frames,
        fps=fps,
        coarse_every_sec=args.coarse_every_sec,
        fine_every_sec=args.fine_every_sec,
        zoom_ratio=args.zoom_ratio,
        final_min_arms=args.final_min_arms,
        final_max_arms=args.final_max_arms,
        min_coarse_segments=args.min_coarse_segments,
        min_zoom_segments=args.min_zoom_segments,
        region_half_window_sec=region_half_window_sec,
        extra_samples_per_region=args.extra_samples_per_region,
        min_variance_threshold=args.min_variance_threshold,
        fine_uniform_ratio=args.fine_uniform_ratio,
        interpolation_method=args.interpolation_method,
        all_coarse_indices=all_coarse_indices,
        all_coarse_similarities=all_coarse_similarities,
        all_fine_indices=[],  # Fine sampling will be handled internally
        all_fine_similarities=[],
        rng=rng,
        k=args.num_keyframes,
        top_ratio=args.top_ratio,
        min_gap_sec=auto_min_gap_sec,
        temperature=args.temperature
    )

    # Prepare sampling details
    sampling_details = prepare_sampling_details(
        all_coarse_indices, all_coarse_similarities,
        [], [],  # Fine sampling details will be empty for now
        arms, selected_frames, total_frames, fps, len(all_coarse_indices),
        arm_selection_probs, arm_selection_counts
    )

    return selected_frames, sampling_details


def prepare_sampling_details(coarse_indices: List[int], coarse_similarities: List[float],
                           fine_indices: List[int], fine_similarities: List[float],
                           arms: List[Dict], selected_frames: List[int],
                           total_frames: int, fps: float, budget_used: int,
                           arm_selection_probs: List[float], arm_selection_counts: List[int]) -> Dict:
    """
    Prepare detailed sampling information for export.
    """
    def create_temporal_order(frame_indices: List[int], similarities: List[float], fps: float) -> List[Dict]:
        """Create temporal order list with timestamps."""
        if not frame_indices:
            return []
        
        combined = list(zip(frame_indices, similarities))
        combined.sort(key=lambda x: x[0])
        
        temporal_order = []
        for frame_idx, score in combined:
            temporal_order.append({
                "frame_idx": int(frame_idx),
                "score": float(score),
                "timestamp": float(frame_idx / max(1.0, fps))
            })
        return temporal_order
    
    # Prepare coarse sampling info
    coarse_sampling = {
        "frame_indices": [int(idx) for idx in coarse_indices],
        "relevance_scores": [float(score) for score in coarse_similarities],
        "temporal_order": create_temporal_order(coarse_indices, coarse_similarities, fps),
        "budget_used": int(len(coarse_indices))
    }
    
    # Prepare fine sampling info
    fine_sampling = {
        "frame_indices": [int(idx) for idx in fine_indices],
        "relevance_scores": [float(score) for score in fine_similarities],
        "temporal_order": create_temporal_order(fine_indices, fine_similarities, fps),
        "budget_used": int(len(fine_indices))
    }
    
    # Prepare arms info
    arms_info = {
        "total_arms": len(arms),
        "frames_per_arm": total_frames // max(1, len(arms)),
        "arms": []
    }
    
    for i, arm in enumerate(arms):
        times_selected = arm_selection_counts[i] if i < len(arm_selection_counts) else 0
        arm_info = {
            "arm_id": int(arm['arm_id']),
            "start_frame": int(arm['start']),
            "end_frame": int(arm['end']),
            "focus_score": float(arm['focus_score']),
            "focus_after_coarse": float(arm['focus_after_coarse']) if arm.get('focus_after_coarse') is not None else None,
            "focus_after_fine": float(arm['focus_after_fine']) if arm.get('focus_after_fine') is not None else None,
            "times_selected": int(times_selected),
            "mean_similarity": float(arm['mean_sim']),
            "variance": float(arm['variance']),
            "samples_count": int(arm['samples'])
        }
        arms_info["arms"].append(arm_info)
    
    # Prepare video metadata
    video_metadata = {
        "total_frames": int(total_frames),
        "fps": float(fps),
        "duration_seconds": float(total_frames / max(1.0, fps)),
        "budget_used": int(budget_used)
    }
    
    return {
        "coarse_sampling": coarse_sampling,
        "fine_sampling": fine_sampling,
        "arms_info": arms_info,
        "arm_selection_probabilities": [float(prob) for prob in arm_selection_probs],
        "final_selected_frames": [int(idx) for idx in selected_frames],
        "video_metadata": video_metadata
    }


# ============================================================================
# Ray Worker Functions
# ============================================================================

@ray.remote(num_gpus=1)
def ray_worker(dp_rank: int, output_json_base_prefix: str, data_slice, args_dict):
    """Ray worker for distributed processing."""
    worker_start_time = time.time()

    class Args: pass
    args = Args()
    for k, v in args_dict.items():
        setattr(args, k, v)

    device = 'cuda:0'
    full_output_dir = os.path.join('./selected_frames', args.dataset_name, args.output_dir)
    os.makedirs(full_output_dir, exist_ok=True)
    output_json = os.path.join(full_output_dir, f"{output_json_base_prefix}_rank{dp_rank}.json")

    model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", args.blip_model,
                                                                       device=device, is_eval=True)

    video_root = (args.dataset_path + '/videos' if args.dataset_name == 'longvideobench' else args.dataset_path + '/data')
    rng = np.random.default_rng(args.seed + dp_rank)

    results = []
    budget_stats = []
    sampling_details_results = []
    
    pbar = tqdm(data_slice, desc=f"Rank {dp_rank}", ncols=100)
    for original_idx, data in pbar:
        try:
            text = data['question']
            video_file = (os.path.join(video_root, data['video_path'])
                          if args.dataset_name == 'longvideobench'
                          else os.path.join(video_root, data['videoID'] + '.mp4'))

            if not os.path.exists(video_file):
                selected = []
                budget_used = 0
                total_frames = 0
                video_duration = 0.0
                sampling_details = {
                    "coarse_sampling": {"frame_indices": [], "relevance_scores": [], "temporal_order": [], "budget_used": 0},
                    "fine_sampling": {"frame_indices": [], "relevance_scores": [], "temporal_order": [], "budget_used": 0},
                    "arms_info": {"total_arms": 0, "frames_per_arm": 0, "arms": []},
                    "arm_selection_probabilities": [],
                    "final_selected_frames": [],
                    "video_metadata": {"total_frames": 0, "fps": 0.0, "duration_seconds": 0.0, "budget_used": 0}
                }
            else:
                vr = VideoReader(video_file, ctx=cpu(0))
                fps = float(vr.get_avg_fps())
                total_frames = len(vr)
                video_duration = float(total_frames) / max(1.0, fps)

                selected, sampling_details = process_single_video(
                    vr, text, vis_processors, text_processors, model, device, args, rng
                )
                budget_used = sampling_details["coarse_sampling"]["budget_used"]

            results.append({"original_idx": original_idx, "selected_frames": [int(x) for x in selected]})
            budget_stats.append({
                "original_idx": original_idx,
                "budget_used": int(budget_used),
                "total_frames": int(total_frames),
                "video_duration": float(video_duration)
            })
            sampling_details_results.append({
                "original_idx": original_idx,
                **sampling_details
            })

            with open(output_json, 'w') as f:
                json.dump(results, f)
            pbar.set_postfix({"processed": len(results), "last_selected": len(selected)})

        except Exception as e:
            print(f"Error on video {original_idx}: {e}")
            results.append({"original_idx": original_idx, "selected_frames": []})
            budget_stats.append({
                "original_idx": original_idx,
                "budget_used": 0,
                "total_frames": 0,
                "video_duration": 0.0
            })
            sampling_details_results.append({
                "original_idx": original_idx,
                "coarse_sampling": {"frame_indices": [], "relevance_scores": [], "temporal_order": [], "budget_used": 0},
                "fine_sampling": {"frame_indices": [], "relevance_scores": [], "temporal_order": [], "budget_used": 0},
                "arms_info": {"total_arms": 0, "frames_per_arm": 0, "arms": []},
                "arm_selection_probabilities": [],
                "final_selected_frames": [],
                "video_metadata": {"total_frames": 0, "fps": 0.0, "duration_seconds": 0.0, "budget_used": 0}
            })
            with open(output_json, 'w') as f:
                json.dump(results, f)

    worker_end_time = time.time()
    worker_runtime_hours = (worker_end_time - worker_start_time) / 3600

    return output_json, budget_stats, worker_runtime_hours, sampling_details_results


# ============================================================================
# File I/O Functions
# ============================================================================

def merge_json_files(output_dir: str, output_json_base_prefix: str, dp_size: int, merged_output_path: str):
    """Merge results from multiple workers."""
    all_results = {}
    for dp_rank in range(dp_size):
        fname = os.path.join(output_dir, f"{output_json_base_prefix}_rank{dp_rank}.json")
        if os.path.exists(fname):
            with open(fname, 'r') as f:
                rank_results = json.load(f)
                for result in rank_results:
                    all_results[result["original_idx"]] = result["selected_frames"]
        else:
            print(f"Warning: File {fname} not found")

    total_videos = max(all_results.keys()) + 1 if all_results else 0
    final_results = []
    for i in range(total_videos):
        if i in all_results:
            final_results.append(all_results[i])
        else:
            final_results.append([])

    with open(merged_output_path, 'w') as f:
        json.dump(final_results, f)
    print(f"Merged results saved to {merged_output_path}")

    for dp_rank in range(dp_size):
        fname = os.path.join(output_dir, f"{output_json_base_prefix}_rank{dp_rank}.json")
        if os.path.exists(fname):
            os.remove(fname)


def merge_sampling_details_files(sampling_details_results: List[List[Dict]], output_dir: str, merged_sampling_details_path: str):
    """Merge sampling details from multiple workers."""
    all_sampling_details = {}
    for worker_details in sampling_details_results:
        for detail in worker_details:
            all_sampling_details[detail["original_idx"]] = detail

    total_videos = max(all_sampling_details.keys()) + 1 if all_sampling_details else 0
    final_sampling_details = []
    for i in range(total_videos):
        if i in all_sampling_details:
            final_sampling_details.append(all_sampling_details[i])
        else:
            final_sampling_details.append({
                "original_idx": i,
                "coarse_sampling": {"frame_indices": [], "relevance_scores": [], "temporal_order": [], "budget_used": 0},
                "fine_sampling": {"frame_indices": [], "relevance_scores": [], "temporal_order": [], "budget_used": 0},
                "arms_info": {"total_arms": 0, "frames_per_arm": 0, "arms": []},
                "arm_selection_probabilities": [],
                "final_selected_frames": [],
                "video_metadata": {"total_frames": 0, "fps": 0.0, "duration_seconds": 0.0, "budget_used": 0}
            })

    with open(merged_sampling_details_path, 'w') as f:
        json.dump(final_sampling_details, f, indent=2)
    print(f"Merged sampling details saved to {merged_sampling_details_path}")


# ============================================================================
# CLI Interface
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Keyframe extraction with FOCUS approach (Frame-Optimistic Confidence Upper-bound Selection)')
    parser.add_argument('--dataset_name', type=str, default='longvideobench',
                        help='support longvideobench and videomme')
    parser.add_argument('--dataset_path', type=str, default='./datasets/longvideobench',
                        help='path to the dataset root')
    parser.add_argument('--output_dir', type=str, default='focus_blip',
                        help='algorithm name folder under ./selected_frames/{dataset_name}/')
    parser.add_argument('--num_keyframes', type=int, default=64,
                        help='number of keyframes to select')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for BLIP processing')
    parser.add_argument('--blip_model', type=str, default='large',
                        help='BLIP model type (base, large)')

    # Hybrid selection parameters
    parser.add_argument('--top_ratio', type=float, default=0.2,
                        help='Ratio to determine top-ranked selection count: top_count = ratio * min(num_keyframes, computed_frames) (0~1)')
    parser.add_argument('--temperature', type=float, default=0.06, help='Softmax temperature for within-arm sampling when using interpolation')
    parser.add_argument('--min_gap_sec', type=float, default=1.0, help='Fixed minimum temporal gap between selections (sec)')

    # Adaptive min-gap controls
    parser.add_argument('--disable_gap_below_sec', type=float, default=0.2,
                        help='Disable min-gap if average spacing <= this (sec)')
    parser.add_argument('--gap_ratio_of_avg', type=float, default=0.25,
                        help='min-gap = min(fixed, ratio * average spacing) when not disabled')

    # Proportional zooming controls (coarse -> fine)
    parser.add_argument('--coarse_every_sec', type=float, default=16.0,
                        help='Coarse level: sample 1 frame every X seconds')
    parser.add_argument('--fine_every_sec', type=float, default=1.0,
                        help='Fine level: sample 1 frame every Y seconds in zoomed regions')
    parser.add_argument('--zoom_ratio', type=float, default=0.25,
                        help='Fraction of coarse segments to zoom into (0~1) and also used in final arm selection')
    parser.add_argument('--min_coarse_segments', type=int, default=8,
                        help='Ensure at least this many coarse segments')
    parser.add_argument('--min_zoom_segments', type=int, default=4,
                        help='Ensure at least this many zoomed regions')
    parser.add_argument('--region_half_window_sec', type=float, default=None,
                        help='Half window size (sec) around each coarse center; default=coarse_every_sec/2')

    # FOCUS shared parameters
    parser.add_argument('--extra_samples_per_region', type=int, default=2,
                        help='Extra random samples per region for initial variance estimation')
    parser.add_argument('--min_variance_threshold', type=float, default=1e-6,
                        help='Minimum variance threshold to avoid division by zero issues in confidence upper-bound')

    # FOCUS specifics
    parser.add_argument('--fine_uniform_ratio', type=float, default=0.5,
                        help='Ratio of uniform sampling in fine sampling stage (0~1). Rest will be random sampling.')
    parser.add_argument('--interpolation_method', type=str, default='nearest', choices=['nearest', 'linear', 'rbf', 'uniform'],
                        help='Interpolation method for estimating scores within arms')
    parser.add_argument('--final_min_arms', type=int, default=4,
                        help='Minimum number of arms to use in final allocation (after zoom_ratio)')
    parser.add_argument('--final_max_arms', type=int, default=32,
                        help='Maximum number of arms to use in final allocation (after zoom_ratio)')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    return parser.parse_args()


def main():
    """Main function for running FOCUS keyframe extraction."""
    args = parse_arguments()

    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)

    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")

    ray.init()
    DP_SIZE = min(8, gpu_count)
    print(f"Using {DP_SIZE} workers")

    time_stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_json_base_prefix = f'keyframe_focus_{args.dataset_name}_{time_stamp}'

    output_dir = os.path.join('./selected_frames', args.dataset_name, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    merged_output_path = os.path.join(output_dir, 'selected_frames.json')
    merged_sampling_details_path = os.path.join(output_dir, 'sampling_details.json')

    if args.dataset_name == 'longvideobench':
        label_path = os.path.join(args.dataset_path, 'lvb_val.json')
    elif args.dataset_name == 'videomme':
        label_path = os.path.join(args.dataset_path, 'videomme.json')
    else:
        raise ValueError('dataset_name: longvideobench or videomme')

    if not os.path.exists(label_path):
        raise OSError('the label file does not exist')
    with open(label_path, 'r') as f:
        datas = json.load(f)
    print(f"Total videos to process: {len(datas)}")

    total = len(datas)
    per_rank = (total + DP_SIZE - 1) // DP_SIZE

    original_indices = list(range(total))
    shuffled_indices = original_indices.copy()
    random.shuffle(shuffled_indices)
    print(f"Shuffled data indices for load balancing across {DP_SIZE} workers")

    args_dict = vars(args)
    ray_tasks = []
    for dp_rank in range(DP_SIZE):
        start = dp_rank * per_rank
        end = min(start + per_rank, total)
        shuffled_slice_indices = shuffled_indices[start:end]
        data_slice = [(orig_idx, datas[orig_idx]) for orig_idx in shuffled_slice_indices]
        if len(data_slice) > 0:
            ray_tasks.append(ray_worker.remote(dp_rank, output_json_base_prefix, data_slice, args_dict))

    print("Processing videos in parallel...")
    ray_results = ray.get(ray_tasks)

    all_budget_stats = []
    all_sampling_details = []
    total_gpu_hours = 0.0
    for _, stats, worker_hours, sampling_details in ray_results:
        all_budget_stats.extend(stats)
        all_sampling_details.append(sampling_details)
        total_gpu_hours += worker_hours

    print("Merging results...")
    merge_json_files(output_dir, output_json_base_prefix, DP_SIZE, merged_output_path)
    print("Merging sampling details...")
    merge_sampling_details_files(all_sampling_details, output_dir, merged_sampling_details_path)

    total_budget_used = sum(s.get('budget_used', 0) for s in all_budget_stats)
    total_frames = sum(s.get('total_frames', 0) for s in all_budget_stats)
    total_duration = sum(s.get('video_duration', 0.0) for s in all_budget_stats)

    frame_speedup = (total_frames / total_budget_used) if total_budget_used > 0 else 0.0
    time_speedup = (total_duration / total_budget_used) if total_budget_used > 0 else 0.0

    print("\n" + "=" * 60)
    print("BUDGET USAGE STATISTICS")
    print("=" * 60)
    print(f"Total videos processed: {len(all_budget_stats)}")
    print(f"Total budget used (BLIP forward passes): {total_budget_used:,}")
    print(f"Total frames in all videos: {total_frames:,}")
    print(f"Total video duration: {total_duration:.1f} seconds ({total_duration/3600:.2f} hours)")
    print(f"  Frame-based speedup: {frame_speedup:.2f}x")
    print(f"  Time-based  speedup: {time_speedup:.2f}x")
    print("=" * 60)
    print("Method: FOCUS (Frame-Optimistic Confidence Upper-bound Selection)")
    print(f"  Extra samples per region: {args.extra_samples_per_region}")
    print(f"  Min variance threshold: {args.min_variance_threshold}")
    print(f"  Fine uniform ratio: {args.fine_uniform_ratio:.2f}")
    print(f"  Interpolation method: {args.interpolation_method}")
    print(f"  Top-ranked ratio: {args.top_ratio:.2f}")
    print(f"  Final selection arms: zoom_ratio={args.zoom_ratio}, bounds=[{args.final_min_arms}, {args.final_max_arms}]")
    print("=" * 60)

    stats_output_path = os.path.join(output_dir, "extraction_stats.json")
    extraction_stats = {
        "gpu_usage": {
            "total_gpu_hours": total_gpu_hours,
            "num_workers": DP_SIZE,
            "avg_gpu_hours_per_worker": (total_gpu_hours / DP_SIZE) if DP_SIZE > 0 else 0.0
        },
        "budget_usage": {
            "total_budget_used": total_budget_used,
            "total_videos_processed": len(all_budget_stats),
            "total_frames": total_frames,
            "total_duration_sec": total_duration,
            "total_duration_hours": total_duration / 3600 if total_duration else 0.0,
            "frame_speedup": frame_speedup,
            "time_speedup": time_speedup
        },
        "algorithm_params": {
            "blip_model": args.blip_model,
            "top_ratio": args.top_ratio,
            "extra_samples_per_region": args.extra_samples_per_region,
            "min_variance_threshold": args.min_variance_threshold,
            "temperature": args.temperature,
            "fine_uniform_ratio": args.fine_uniform_ratio,
            "interpolation_method": args.interpolation_method,
            "final_min_arms": args.final_min_arms,
            "final_max_arms": args.final_max_arms
        },
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "output_path": merged_output_path,
        "experiment_name": args.output_dir
    }
    with open(stats_output_path, 'w') as f:
        json.dump(extraction_stats, f, indent=2)

    print(f"\nExtraction statistics saved to: {stats_output_path}")
    print(f"\nFOCUS keyframe extraction completed. Results saved to {merged_output_path}")
    ray.shutdown()


if __name__ == '__main__':
    main()
