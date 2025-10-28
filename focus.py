"""
FOCUS: Frame-Optimistic Confidence Upper-bound Selection

Core algorithm implementation for keyframe extraction using confidence upper-bound bandit approach.
This module contains only the algorithm logic, without data processing or I/O operations.
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from scipy.interpolate import Rbf


# ============================================================================
# Core FOCUS Algorithm Functions
# ============================================================================

def setup_arms(total_frames: int, coarse_stride: int) -> List[Dict]:
    """
    Setup arms: Partition video frames into temporal arms.
    
    Args:
        total_frames: Total number of frames in the video
        coarse_stride: Stride for coarse sampling
        
    Returns:
        List of arm dictionaries with metadata
    """
    num_regions = max(1, total_frames // coarse_stride)
    region_size = max(1, total_frames // num_regions)
    
    arms = []
    for region_idx in range(num_regions):
        region_start = region_idx * region_size
        region_end = min((region_idx + 1) * region_size, total_frames)
        if region_end > region_start:
            arms.append({
                'arm_id': region_idx,
                'start': region_start,
                'end': region_end - 1,  # inclusive end
                'samples': 0,
                'total_reward': 0.0,
                'sum_squared_reward': 0.0,
                'mean_sim': 0.0,
                'variance': 0.0,
                'focus_score': 0.0,
                'focus_after_coarse': None,
                'focus_after_fine': None,
                'sampled_indices': [],
                'sampled_scores': []
            })
    return arms


def coarse_sampling_in_arms(arms: List[Dict], extra_samples_per_region: int, 
                           rng: np.random.Generator) -> List[int]:
    """
    Coarse Sampling in each arm.
    
    Args:
        arms: List of arm dictionaries
        extra_samples_per_region: Number of extra random samples per region
        rng: Random number generator
        
    Returns:
        List of frame indices to sample
    """
    all_coarse_indices = []
    for arm in arms:
        start, end = arm['start'], arm['end']
        # Sample center point
        center_idx = (start + end) // 2
        sampled_indices = [center_idx]
        
        # Sample extra random points in this arm
        available_indices = list(range(start, end + 1))
        if center_idx in available_indices:
            available_indices.remove(center_idx)
        
        if len(available_indices) > 0:
            extra_count = min(extra_samples_per_region, len(available_indices))
            extra_indices = rng.choice(available_indices, size=extra_count, replace=False)
            sampled_indices.extend(extra_indices.tolist())
        
        arm['sampled_indices'] = sampled_indices
        all_coarse_indices.extend(sampled_indices)
    
    return all_coarse_indices


def update_arms_with_scores(arms: List[Dict], all_indices: List[int], 
                           all_similarities: List[float]) -> None:
    """
    Update arms with new sampling results and compute statistics.
    
    Args:
        arms: List of arm dictionaries
        all_indices: List of sampled frame indices
        all_similarities: List of corresponding similarity scores
    """
    # Create mapping from index to score
    idx_to_score = {idx: score for idx, score in zip(all_indices, all_similarities)}
    
    for arm in arms:
        # Update with new scores
        new_scores = []
        for idx in arm['sampled_indices']:
            if idx in idx_to_score:
                score = idx_to_score[idx]
                new_scores.append(score)
                if idx not in [s[0] for s in arm['sampled_scores']]:  # avoid duplicates
                    arm['sampled_scores'].append((idx, score))
        
        # Recompute statistics for this arm
        if arm['sampled_scores']:
            all_arm_scores = [score for _, score in arm['sampled_scores']]
            arm['samples'] = len(all_arm_scores)
            arm['mean_sim'] = float(np.mean(all_arm_scores))
            arm['variance'] = float(np.var(all_arm_scores)) if len(all_arm_scores) > 1 else 0.0


def update_focus_scores_for_arms(arms: List[Dict], min_variance_threshold: float) -> None:
    """
    Update FOCUS confidence upper-bound scores for all arms.
    
    Args:
        arms: List of arm dictionaries
        min_variance_threshold: Minimum variance threshold to avoid division by zero
    """
    total_samples = sum(arm['samples'] for arm in arms)
    
    for arm in arms:
        n_i = arm['samples']
        mean = arm['mean_sim']
        var = max(arm['variance'], min_variance_threshold)
        
        focus_score = mean
        if total_samples > 1 and n_i > 0:
            focus_score += math.sqrt(max(0.0, 2 * math.log(total_samples) * var / n_i))
            focus_score += (3 * math.log(total_samples) / n_i)
        
        arm['focus_score'] = focus_score


def choose_promising_arms(arms: List[Dict], zoom_ratio: float, 
                         min_zoom_segments: int) -> List[Dict]:
    """
    Choose promising arms using FOCUS confidence upper-bound scores.
    
    Args:
        arms: List of arm dictionaries
        zoom_ratio: Fraction of arms to select for fine sampling
        min_zoom_segments: Minimum number of segments to zoom into
        
    Returns:
        List of selected promising arms
    """
    arms_sorted = sorted(arms, key=lambda x: x['focus_score'], reverse=True)
    zoom_count = max(min_zoom_segments, int(np.ceil(len(arms) * zoom_ratio)))
    zoom_count = min(zoom_count, len(arms))
    
    selected_arms = arms_sorted[:zoom_count]
    return selected_arms


def fine_sampling_in_arms(selected_arms: List[Dict], total_frames: int, 
                         fine_stride: int, region_half_window: int,
                         coarse_indices_set: set, fine_uniform_ratio: float,
                         rng: np.random.Generator) -> List[int]:
    """
    Fine sampling in promising arms with mixed uniform/random strategy.
    
    Args:
        selected_arms: List of selected promising arms
        total_frames: Total number of frames in video
        fine_stride: Stride for fine sampling
        region_half_window: Half window size around arm centers
        coarse_indices_set: Set of already sampled coarse indices
        fine_uniform_ratio: Ratio of uniform sampling (0~1)
        rng: Random number generator
        
    Returns:
        List of frame indices for fine sampling
    """
    candidate_fine = set()
    
    for arm in selected_arms:
        arm_center = (arm['start'] + arm['end']) // 2
        start = max(0, arm_center - region_half_window)
        end = min(total_frames - 1, arm_center + region_half_window)
        
        window_size = end - start + 1
        required_samples = max(1, window_size // fine_stride)
        
        # Split into uniform and random sampling
        uniform_count = int(round(required_samples * fine_uniform_ratio))
        random_count = required_samples - uniform_count
        
        # Uniform sampling
        if uniform_count > 0:
            if window_size <= uniform_count:
                # If window is small, just take center
                center_idx = (start + end) // 2
                if center_idx not in coarse_indices_set and center_idx not in candidate_fine:
                    candidate_fine.add(center_idx)
            else:
                # Uniform spacing
                interval_size = window_size / uniform_count
                for i in range(uniform_count):
                    uniform_idx = start + int(i * interval_size + interval_size / 2)
                    uniform_idx = min(uniform_idx, end)
                    if uniform_idx not in coarse_indices_set and uniform_idx not in candidate_fine:
                        candidate_fine.add(uniform_idx)
        
        # Random sampling for remaining slots
        if random_count > 0:
            available_indices = [i for i in range(start, end + 1) 
                               if i not in coarse_indices_set and i not in candidate_fine]
            if available_indices:
                actual_random_count = min(random_count, len(available_indices))
                random_indices = rng.choice(available_indices, size=actual_random_count, replace=False)
                for idx in random_indices:
                    candidate_fine.add(int(idx))
    
    return sorted(list(candidate_fine))


def estimate_arm_scores(sampled_indices: List[int], sampled_scores: List[float],
                       arm_start: int, arm_end: int, 
                       interpolation_method: str = 'nearest') -> Dict[int, float]:
    """
    Estimate similarity scores for all frames within a single arm using various interpolation methods.
    
    Args:
        sampled_indices: List of sampled frame indices
        sampled_scores: List of corresponding similarity scores
        arm_start: Start frame index of the arm
        arm_end: End frame index of the arm
        interpolation_method: Method for interpolation ('nearest', 'linear', 'rbf')
        
    Returns:
        Dictionary mapping frame indices to estimated scores
    """
    if len(sampled_indices) < 1 or not sampled_scores:
        return {}
    
    # Create candidate indices for all frames within this arm
    arm_candidates = list(range(arm_start, arm_end + 1))
    if not arm_candidates:
        return {}
    
    sampled_indices_arr = np.array(sampled_indices, dtype=float)
    sampled_scores_arr = np.array(sampled_scores, dtype=float)
    arm_candidates_arr = np.array(arm_candidates, dtype=float)
    
    try:
        if interpolation_method == 'nearest':
            # Nearest neighbor interpolation
            estimated_scores = []
            for candidate in arm_candidates_arr:
                nearest_idx = np.argmin(np.abs(sampled_indices_arr - candidate))
                estimated_scores.append(sampled_scores_arr[nearest_idx])
            estimated_scores_arr = np.array(estimated_scores)
            
        elif interpolation_method == 'linear':
            # Linear interpolation
            if len(sampled_indices) >= 2:
                estimated_scores_arr = np.interp(arm_candidates_arr, sampled_indices_arr, sampled_scores_arr)
            else:
                # Fallback to constant value if only one sample
                estimated_scores_arr = np.full_like(arm_candidates_arr, sampled_scores_arr[0])
            
        elif interpolation_method == 'rbf':
            # RBF interpolation using scipy with automatic parameters
            if len(sampled_indices) >= 2:
                rbf = Rbf(sampled_indices_arr, sampled_scores_arr, function='gaussian', smooth=0.0)
                estimated_scores_arr = rbf(arm_candidates_arr)
            else:
                # Fallback to constant value if only one sample
                estimated_scores_arr = np.full_like(arm_candidates_arr, sampled_scores_arr[0])
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation_method}")
        
        # Clamp results to reasonable range
        estimated_scores_arr = np.clip(estimated_scores_arr, 0.0, 1.0)
        
        result = {}
        for candidate_idx, estimated_score in zip(arm_candidates, estimated_scores_arr):
            result[candidate_idx] = float(estimated_score)
        return result
        
    except Exception as e:
        print(f"Interpolation failed in arm [{arm_start}, {arm_end}] with method {interpolation_method}: {e}")
        # Fallback: use mean score for all candidates
        mean_score = np.mean(sampled_scores)
        result = {}
        for candidate_idx in arm_candidates:
            result[candidate_idx] = float(mean_score)
        return result


def choose_top_frames_from_sampled(all_sampled_scores: Dict[int, float], 
                                  k: int, top_ratio: float) -> List[int]:
    """
    Choose top_ratio frames from all sampled scores.
    
    Args:
        all_sampled_scores: Dictionary mapping frame indices to similarity scores
        k: Total number of keyframes to select
        top_ratio: Ratio of top-ranked frames to select
        
    Returns:
        List of selected top frame indices
    """
    if not all_sampled_scores or k <= 0:
        return []
    
    num_computed_frames = len(all_sampled_scores)
    k_top = int(round(top_ratio * min(k, num_computed_frames)))
    
    sorted_sampled = sorted(all_sampled_scores.items(), key=lambda x: x[1], reverse=True)
    top_frames = [idx for idx, _ in sorted_sampled[:k_top]]
    
    return top_frames


def sample_remaining_frames_by_focus_fixed_top(
    arms: List[Dict],
    remaining_count: int,
    all_sampled_scores: Dict[int, float],
    selected_frames: List[int],
    interpolation_method: str,
    min_gap_sec: float,
    fps: float,
    temperature: float,
    zoom_ratio: float,
    rng: np.random.Generator,
    min_arms: int,
    max_arms: int,
) -> Tuple[List[int], List[float], List[int]]:
    """
    FOCUS change: Use zoom_ratio to pick top arms for the final allocation with hard bounds.
      - S = clamp(ceil(len(arms) * zoom_ratio), min_arms, max_arms, <= len(arms))
      - Evenly distribute remaining_count across these S arms
      - Sampling within an arm follows the same rules as v3 (uniform or interpolation-based)
    
    Args:
        arms: List of arm dictionaries
        remaining_count: Number of remaining frames to select
        all_sampled_scores: Dictionary of all sampled scores
        selected_frames: List of already selected frames
        interpolation_method: Method for interpolation within arms
        min_gap_sec: Minimum temporal gap between selections
        fps: Video frame rate
        temperature: Temperature for softmax sampling
        zoom_ratio: Ratio for selecting top arms
        rng: Random number generator
        min_arms: Minimum number of arms to use
        max_arms: Maximum number of arms to use
        
    Returns:
        Tuple of (new_frames, arm_selection_probs, arm_selection_counts)
    """
    if remaining_count <= 0 or not arms:
        return [], [], []

    gap_frames = int(round(min_gap_sec * fps)) if min_gap_sec > 0 else 0
    selected_set = set(selected_frames)

    def respects_gap(cand: int, chosen_set: set) -> bool:
        if gap_frames <= 0:
            return True
        for frame in chosen_set:
            if abs(cand - frame) < gap_frames:
                return False
        return True

    # Compute number of top arms using zoom_ratio with hard bounds
    total_arms = len(arms)
    S = int(np.ceil(total_arms * max(0.0, min(1.0, float(zoom_ratio)))))
    S = max(min_arms, S)
    S = min(S, max_arms)
    S = min(S, total_arms)

    arms_sorted = sorted([(i, arm) for i, arm in enumerate(arms)], key=lambda x: x[1]['focus_score'], reverse=True)
    top_arm_entries = arms_sorted[:S]

    # Even allocation
    base = remaining_count // S
    rem = remaining_count % S
    per_arm_need = [base + (1 if i < rem else 0) for i in range(S)]

    arm_selection_counts = [0 for _ in range(total_arms)]
    new_frames = []
    current_selected_set = selected_set.copy()

    for rank, ((arm_idx, arm)) in enumerate(top_arm_entries):
        needed_count = per_arm_need[rank]
        if needed_count == 0:
            continue

        arm_start, arm_end = arm['start'], arm['end']
        all_arm_candidates = list(range(arm_start, arm_end + 1))
        available_candidates = [c for c in all_arm_candidates if c not in current_selected_set]
        if not available_candidates:
            continue

        if interpolation_method == 'uniform':
            # Random sampling within arm
            if gap_frames > 0:
                gap_candidates = [c for c in available_candidates if respects_gap(c, current_selected_set)]
                candidates_to_use = gap_candidates if gap_candidates else available_candidates
            else:
                candidates_to_use = available_candidates

            actual_count = min(needed_count, len(candidates_to_use))
            if actual_count > 0:
                sampled_indices = rng.choice(candidates_to_use, size=actual_count, replace=False)
                for idx in sampled_indices:
                    new_frames.append(int(idx))
                    current_selected_set.add(int(idx))
                    arm_selection_counts[arm_idx] += 1
        else:
            # Interpolation-based sampling within arm
            arm_sampled_indices = [idx for idx, _ in arm['sampled_scores']]
            arm_sampled_similarities = [score for _, score in arm['sampled_scores']]

            arm_estimated_scores = estimate_arm_scores(
                arm_sampled_indices, arm_sampled_similarities,
                arm_start, arm_end, interpolation_method
            )

            scored_candidates = []
            for cand in available_candidates:
                if cand in arm_estimated_scores:
                    score = arm_estimated_scores[cand]
                    if gap_frames == 0 or respects_gap(cand, current_selected_set):
                        scored_candidates.append((cand, score))

            if not scored_candidates:
                # Fallback: ignore gap if necessary
                for cand in available_candidates:
                    if cand in arm_estimated_scores:
                        scored_candidates.append((cand, arm_estimated_scores[cand]))

            if scored_candidates:
                candidates, scores = zip(*scored_candidates)
                candidates = list(candidates)
                scores = np.array(scores, dtype=np.float64)

                if scores.max() > scores.min():
                    scores = (scores - scores.min()) / (scores.max() - scores.min())
                else:
                    scores = np.ones_like(scores)

                logits = scores / max(1e-12, temperature)
                logits = logits - logits.max()
                probs = np.exp(logits)
                probs = probs / probs.sum()

                actual_count = min(needed_count, len(candidates))
                if actual_count > 0:
                    sampled_positions = rng.choice(len(candidates), size=actual_count, p=probs, replace=False)
                    for pos in sampled_positions:
                        idx = candidates[pos]
                        new_frames.append(int(idx))
                        current_selected_set.add(int(idx))
                        arm_selection_counts[arm_idx] += 1
            else:
                candidates_to_use = available_candidates
                if gap_frames > 0:
                    gap_candidates = [c for c in available_candidates if respects_gap(c, current_selected_set)]
                    candidates_to_use = gap_candidates if gap_candidates else available_candidates
                actual_count = min(needed_count, len(candidates_to_use))
                if actual_count > 0:
                    sampled_indices = rng.choice(candidates_to_use, size=actual_count, replace=False)
                    for idx in sampled_indices:
                        new_frames.append(int(idx))
                        current_selected_set.add(int(idx))
                        arm_selection_counts[arm_idx] += 1

    # Derive per-arm probabilities from counts
    if remaining_count > 0:
        arm_selection_probs = [c / remaining_count for c in arm_selection_counts]
    else:
        arm_selection_probs = [0.0 for _ in arm_selection_counts]

    return new_frames, arm_selection_probs, arm_selection_counts


# ============================================================================
# Main FOCUS Algorithm
# ============================================================================

def run_focus_algorithm(
    total_frames: int,
    fps: float,
    coarse_every_sec: float,
    fine_every_sec: float,
    zoom_ratio: float,
    final_min_arms: int,
    final_max_arms: int,
    min_coarse_segments: int,
    min_zoom_segments: int,
    region_half_window_sec: Optional[float],
    extra_samples_per_region: int,
    min_variance_threshold: float,
    fine_uniform_ratio: float,
    interpolation_method: str,
    all_coarse_indices: List[int],
    all_coarse_similarities: List[float],
    all_fine_indices: List[int],
    all_fine_similarities: List[float],
    rng: np.random.Generator,
    k: Optional[int] = None,
    top_ratio: float = 0.2,
    min_gap_sec: float = 0.0,
    temperature: float = 0.06
) -> Tuple[List[int], List[Dict], List[float], List[int]]:
    """
    Main FOCUS algorithm for keyframe selection.
    
    Args:
        total_frames: Total number of frames in video
        fps: Video frame rate
        coarse_every_sec: Coarse sampling interval in seconds
        fine_every_sec: Fine sampling interval in seconds
        zoom_ratio: Ratio for selecting promising arms
        final_min_arms: Minimum number of arms for final selection
        final_max_arms: Maximum number of arms for final selection
        min_coarse_segments: Minimum number of coarse segments
        min_zoom_segments: Minimum number of zoomed segments
        region_half_window_sec: Half window size for fine sampling
        extra_samples_per_region: Extra samples per region
        min_variance_threshold: Minimum variance threshold
        fine_uniform_ratio: Ratio of uniform sampling in fine stage
        interpolation_method: Method for interpolation
        all_coarse_indices: List of coarse sampling frame indices
        all_coarse_similarities: List of coarse sampling similarity scores
        all_fine_indices: List of fine sampling frame indices
        all_fine_similarities: List of fine sampling similarity scores
        rng: Random number generator
        k: Number of keyframes to select
        top_ratio: Ratio of top-ranked frames
        min_gap_sec: Minimum gap between selections
        temperature: Temperature for softmax sampling
        
    Returns:
        Tuple of (selected_frames, arms, arm_selection_probs, arm_selection_counts)
    """
    video_duration = float(total_frames) / max(1.0, fps)

    coarse_stride = max(1, int(round(coarse_every_sec * fps)))
    desired_coarse = max(min_coarse_segments, int(np.ceil(video_duration / max(1e-6, coarse_every_sec))))
    if desired_coarse > 0:
        coarse_stride = max(1, int(np.floor(total_frames / desired_coarse)))

    fine_stride = max(1, int(round(fine_every_sec * fps)))
    fine_stride = min(fine_stride, coarse_stride)

    if region_half_window_sec is None:
        effective_coarse_sec = max(1e-6, video_duration / max(1, desired_coarse))
        region_half_window_sec = max(1.0 / fps, effective_coarse_sec / 2.0)
    region_half_window = max(1, int(round(region_half_window_sec * fps)))

    if k is None:
        k = 0

    # 0. Setup arms
    arms = setup_arms(total_frames, coarse_stride)

    # 1. Explore and Exploit
    # 1.2 Update arms with coarse results and FOCUS Scores
    update_arms_with_scores(arms, all_coarse_indices, all_coarse_similarities)
    update_focus_scores_for_arms(arms, min_variance_threshold)
    for arm in arms:
        arm['focus_after_coarse'] = float(arm['focus_score'])

    # 1.3 Choose promising arms using FOCUS scores
    selected_arms = choose_promising_arms(arms, zoom_ratio, min_zoom_segments)

    # 1.4 Fine sampling in promising arms with mixed strategy
    coarse_indices_set = set(all_coarse_indices)
    fine_indices = fine_sampling_in_arms(selected_arms, total_frames, fine_stride,
                                        region_half_window, coarse_indices_set,
                                        fine_uniform_ratio, rng)

    # 1.5 Update arms with fine results and FOCUS Scores
    if all_fine_indices:
        update_arms_with_scores(arms, all_fine_indices, all_fine_similarities)
        update_focus_scores_for_arms(arms, min_variance_threshold)
        for arm in arms:
            arm['focus_after_fine'] = float(arm['focus_score'])

    # Merge sampled scores
    all_sampled_scores = {idx: sim for idx, sim in zip(all_coarse_indices, all_coarse_similarities)}
    if all_fine_indices:
        all_sampled_scores.update({idx: sim for idx, sim in zip(all_fine_indices, all_fine_similarities)})

    # 2. Final keyframe selection
    selected_frames = []
    if all_sampled_scores and k > 0:
        # 2.1 Choose top frames from all sampled scores
        selected_frames = choose_top_frames_from_sampled(all_sampled_scores, k, top_ratio)

    # 2.2 Choose remaining frames using FOCUS arm selection rule (zoom_ratio with bounds)
    remaining_count = max(0, k - len(selected_frames))
    arm_selection_probs = []
    arm_selection_counts = []
    if remaining_count > 0:
        additional_frames, arm_selection_probs, arm_selection_counts = sample_remaining_frames_by_focus_fixed_top(
            arms=arms,
            remaining_count=remaining_count,
            all_sampled_scores=all_sampled_scores,
            selected_frames=selected_frames,
            interpolation_method=interpolation_method,
            min_gap_sec=min_gap_sec,
            fps=fps,
            temperature=temperature,
            zoom_ratio=zoom_ratio,
            rng=rng,
            min_arms=final_min_arms,
            max_arms=final_max_arms,
        )
        selected_frames.extend(additional_frames)

    # 2.3 Merge and finalize
    selected_frames = sorted(list(dict.fromkeys(selected_frames)))[:k] if k > 0 else sorted(list(dict.fromkeys(selected_frames)))

    return selected_frames, arms, arm_selection_probs, arm_selection_counts