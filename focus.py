"""
FOCUS: Frame-Optimistic Confidence Upper-bound Selection

Core algorithm implementation for keyframe extraction using confidence upper-bound bandit approach.
This module contains only the algorithm logic, without data processing or I/O operations.
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Callable
from scipy.interpolate import Rbf
from decord import VideoReader


# ============================================================================
# Module-level Helper Functions (Pure Mathematical Operations)
# ============================================================================

def estimate_arm_scores(
    sampled_indices: List[int],
    sampled_scores: List[float],
    arm_start: int,
    arm_end: int,
    interpolation_method: str = 'nearest'
) -> Dict[int, float]:
    """
    Estimate similarity scores for all frames within a single arm using interpolation.
    
    This is a pure function that performs mathematical interpolation without side effects.
    
    Args:
        sampled_indices: List of sampled frame indices within the arm
        sampled_scores: List of corresponding similarity scores
        arm_start: Start frame index of the arm (inclusive)
        arm_end: End frame index of the arm (inclusive)
        interpolation_method: Method for interpolation ('nearest', 'linear', 'rbf', 'uniform')
        
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
            # RBF interpolation using scipy
            if len(sampled_indices) >= 2:
                rbf = Rbf(sampled_indices_arr, sampled_scores_arr, function='gaussian', smooth=0.0)
                estimated_scores_arr = rbf(arm_candidates_arr)
            else:
                # Fallback to constant value if only one sample
                estimated_scores_arr = np.full_like(arm_candidates_arr, sampled_scores_arr[0])
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation_method}")
        
        # Clamp results to reasonable range [0, 1]
        estimated_scores_arr = np.clip(estimated_scores_arr, 0.0, 1.0)
        
        # Convert to dictionary
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


# ============================================================================
# FOCUS Algorithm Class
# ============================================================================

class FOCUS:
    """
    FOCUS: Frame-Optimistic Confidence Upper-bound Selection
    
    A bandit-based keyframe extraction algorithm that uses confidence upper bounds
    to balance exploration and exploitation across temporal segments of the video.
    
    The algorithm operates in several stages:
    1. Setup arms: Partition video into temporal segments
    2. Coarse sampling: Sample center + random points in each arm
    3. Fine sampling: Densely sample in promising arms based on FOCUS scores
    4. Final selection: Combine top-ranked frames with FOCUS-guided arm allocation
    """
    
    def __init__(
        self,
        similarity_fn: Callable[[VideoReader, str, List[int]], List[float]],
        coarse_every_sec: float = 16.0,
        fine_every_sec: float = 1.0,
        zoom_ratio: float = 0.25,
        final_min_arms: int = 4,
        final_max_arms: int = 32,
        min_coarse_segments: int = 8,
        min_zoom_segments: int = 4,
        extra_samples_per_region: int = 2,
        min_variance_threshold: float = 1e-6,
        fine_uniform_ratio: float = 0.5,
        interpolation_method: str = 'nearest',
        top_ratio: float = 0.2,
        temperature: float = 0.06,
        region_half_window_sec: Optional[float] = None
    ):
        """
        Initialize FOCUS keyframe selector.
        
        Args:
            similarity_fn: Function with signature (video, query, frame_indices) -> similarity_scores
            coarse_every_sec: Coarse level sampling interval in seconds
            fine_every_sec: Fine level sampling interval in seconds
            zoom_ratio: Fraction of arms to select for fine sampling and final selection
            final_min_arms: Minimum number of arms to use in final allocation
            final_max_arms: Maximum number of arms to use in final allocation
            min_coarse_segments: Minimum number of coarse segments
            min_zoom_segments: Minimum number of segments to zoom into for fine sampling
            extra_samples_per_region: Extra random samples per region for variance estimation
            min_variance_threshold: Minimum variance threshold for confidence bounds
            fine_uniform_ratio: Ratio of uniform sampling in fine stage (0~1)
            interpolation_method: Method for score interpolation ('nearest', 'linear', 'rbf', 'uniform')
            top_ratio: Ratio of top-ranked frames to select directly
            temperature: Temperature for softmax sampling within arms
            region_half_window_sec: Half window size for fine sampling (defaults to coarse_every_sec/2)
        """
        self.similarity_fn = similarity_fn
        self.coarse_every_sec = coarse_every_sec
        self.fine_every_sec = fine_every_sec
        self.zoom_ratio = zoom_ratio
        self.final_min_arms = final_min_arms
        self.final_max_arms = final_max_arms
        self.min_coarse_segments = min_coarse_segments
        self.min_zoom_segments = min_zoom_segments
        self.extra_samples_per_region = extra_samples_per_region
        self.min_variance_threshold = min_variance_threshold
        self.fine_uniform_ratio = fine_uniform_ratio
        self.interpolation_method = interpolation_method
        self.top_ratio = top_ratio
        self.temperature = temperature
        self.region_half_window_sec = region_half_window_sec
    
    # ========================================================================
    # Public Interface
    # ========================================================================
    
    def select_keyframes(
        self,
        video: VideoReader,
        query: str,
        k: int,
        min_gap_sec: float = 0.0,
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[List[int], Dict]:
        """
        Select keyframes from video based on query using FOCUS algorithm.
        
        Args:
            video: VideoReader object
            query: Query text
            k: Number of keyframes to select
            min_gap_sec: Minimum temporal gap between selections in seconds
            rng: Random number generator (optional, will create default if None)
            
        Returns:
            Tuple of (selected_frames, sampling_details)
                - selected_frames: List of selected frame indices (sorted)
                - sampling_details: Dictionary containing detailed sampling information
        """
        # Extract video metadata
        total_frames = len(video)
        fps = float(video.get_avg_fps())
        video_duration = float(total_frames) / max(1.0, fps)
        
        if rng is None:
            rng = np.random.default_rng()
        
        # Calculate strides and parameters
        coarse_stride = max(1, int(round(self.coarse_every_sec * fps)))
        desired_coarse = max(self.min_coarse_segments, int(np.ceil(video_duration / max(1e-6, self.coarse_every_sec))))
        if desired_coarse > 0:
            coarse_stride = max(1, int(np.floor(total_frames / desired_coarse)))
        
        fine_stride = max(1, int(round(self.fine_every_sec * fps)))
        fine_stride = min(fine_stride, coarse_stride)
        
        if self.region_half_window_sec is None:
            effective_coarse_sec = max(1e-6, video_duration / max(1, desired_coarse))
            region_half_window_sec = max(1.0 / fps, effective_coarse_sec / 2.0)
        else:
            region_half_window_sec = self.region_half_window_sec
        region_half_window = max(1, int(round(region_half_window_sec * fps)))
        
        # Stage 0: Setup arms (temporal partitioning)
        arms = self._setup_arms(total_frames, coarse_stride)
        
        # Stage 1: Coarse sampling
        all_coarse_indices = self._coarse_sampling_in_arms(arms, rng)
        all_coarse_similarities = self.similarity_fn(video, query, all_coarse_indices)
        
        # Update arms with coarse results
        self._update_arms_with_scores(arms, all_coarse_indices, all_coarse_similarities)
        self._update_focus_scores(arms)
        for arm in arms:
            arm['focus_after_coarse'] = float(arm['focus_score'])
        
        # Stage 2: Choose promising arms for fine sampling
        selected_arms = self._choose_promising_arms(arms)
        
        # Stage 3: Fine sampling in promising arms
        coarse_indices_set = set(all_coarse_indices)
        fine_indices = self._fine_sampling_in_arms(
            selected_arms, total_frames, fine_stride,
            region_half_window, coarse_indices_set, rng
        )
        
        # Compute similarities for fine samples
        all_fine_indices = []
        all_fine_similarities = []
        if fine_indices:
            all_fine_indices = fine_indices
            all_fine_similarities = self.similarity_fn(video, query, fine_indices)
            
            # Update arms with fine results
            self._update_arms_with_scores(arms, all_fine_indices, all_fine_similarities)
            self._update_focus_scores(arms)
            for arm in arms:
                arm['focus_after_fine'] = float(arm['focus_score'])
        
        # Merge all sampled scores
        all_sampled_scores = {idx: sim for idx, sim in zip(all_coarse_indices, all_coarse_similarities)}
        if all_fine_indices:
            all_sampled_scores.update({idx: sim for idx, sim in zip(all_fine_indices, all_fine_similarities)})
        
        # Stage 4: Final keyframe selection
        selected_frames = []
        arm_selection_probs = []
        arm_selection_counts = []
        
        if all_sampled_scores and k > 0:
            # 4a. Select top-ranked frames
            selected_frames = self._select_top_frames(all_sampled_scores, k)
            
            # 4b. Select remaining frames using FOCUS arm selection
            remaining_count = max(0, k - len(selected_frames))
            if remaining_count > 0:
                additional_frames, arm_selection_probs, arm_selection_counts = self._select_remaining_frames(
                    arms=arms,
                    remaining_count=remaining_count,
                    all_sampled_scores=all_sampled_scores,
                    selected_frames=selected_frames,
                    fps=fps,
                    min_gap_sec=min_gap_sec,
                    rng=rng
                )
                selected_frames.extend(additional_frames)
        
        # Finalize: deduplicate and sort
        selected_frames = sorted(list(dict.fromkeys(selected_frames)))[:k] if k > 0 else sorted(list(dict.fromkeys(selected_frames)))
        
        # Prepare sampling details
        sampling_details = self._prepare_sampling_details(
            all_coarse_indices, all_coarse_similarities,
            all_fine_indices, all_fine_similarities,
            arms, selected_frames, total_frames, fps,
            len(all_coarse_indices) + len(all_fine_indices),
            arm_selection_probs, arm_selection_counts
        )
        
        return selected_frames, sampling_details
    
    # ========================================================================
    # Internal Methods (Algorithm Stages)
    # ========================================================================
    
    def _setup_arms(self, total_frames: int, coarse_stride: int) -> List[Dict]:
        """
        Stage 0: Setup arms (temporal partitioning).
        
        Partition the video into temporal segments (arms). Each arm represents
        a temporal region that can be independently sampled and evaluated.
        
        Args:
            total_frames: Total number of frames in the video
            coarse_stride: Stride for coarse sampling (determines arm size)
            
        Returns:
            List of arm dictionaries, each containing:
                - arm_id: Unique identifier
                - start, end: Frame range (inclusive)
                - samples, mean_sim, variance: Statistics
                - focus_score: Confidence upper-bound score
                - sampled_indices, sampled_scores: Tracking of samples
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
    
    def _coarse_sampling_in_arms(
        self,
        arms: List[Dict],
        rng: np.random.Generator
    ) -> List[int]:
        """
        Stage 1a: Coarse sampling in each arm.
        
        Sample the center point of each arm plus extra random points for
        variance estimation. This provides initial exploration of all arms.
        
        Args:
            arms: List of arm dictionaries
            rng: Random number generator
            
        Returns:
            List of frame indices to sample (will be passed to similarity_fn)
        """
        all_coarse_indices = []
        for arm in arms:
            start, end = arm['start'], arm['end']
            # Sample center point
            center_idx = (start + end) // 2
            sampled_indices = [center_idx]
            
            # Sample extra random points in this arm for variance estimation
            available_indices = list(range(start, end + 1))
            if center_idx in available_indices:
                available_indices.remove(center_idx)
            
            if len(available_indices) > 0:
                extra_count = min(self.extra_samples_per_region, len(available_indices))
                extra_indices = rng.choice(available_indices, size=extra_count, replace=False)
                sampled_indices.extend(extra_indices.tolist())
            
            arm['sampled_indices'] = sampled_indices
            all_coarse_indices.extend(sampled_indices)
        
        return all_coarse_indices
    
    def _update_arms_with_scores(
        self,
        arms: List[Dict],
        indices: List[int],
        similarities: List[float]
    ) -> None:
        """
        Update arms with new sampling results and compute statistics.
        
        This method updates each arm's sampled_scores and recomputes
        the mean similarity and variance based on all samples collected so far.
        
        Args:
            arms: List of arm dictionaries
            indices: List of sampled frame indices
            similarities: List of corresponding similarity scores
        """
        # Create mapping from index to score
        idx_to_score = {idx: score for idx, score in zip(indices, similarities)}
        
        for arm in arms:
            # Update with new scores (avoid duplicates)
            for idx in arm['sampled_indices']:
                if idx in idx_to_score:
                    score = idx_to_score[idx]
                    if idx not in [s[0] for s in arm['sampled_scores']]:
                        arm['sampled_scores'].append((idx, score))
            
            # Recompute statistics for this arm
            if arm['sampled_scores']:
                all_arm_scores = [score for _, score in arm['sampled_scores']]
                arm['samples'] = len(all_arm_scores)
                arm['mean_sim'] = float(np.mean(all_arm_scores))
                arm['variance'] = float(np.var(all_arm_scores)) if len(all_arm_scores) > 1 else 0.0
    
    def _update_focus_scores(self, arms: List[Dict]) -> None:
        """
        Stage 1b: Compute FOCUS confidence upper-bound scores for all arms.
        
        FOCUS score combines:
        - Mean similarity (exploitation)
        - Variance-based uncertainty term (exploration)
        - Sample count penalty (exploration)
        
        Formula: mean + sqrt(2*log(N)*var/n) + 3*log(N)/n
        where N = total samples, n = samples in this arm, var = variance
        
        Args:
            arms: List of arm dictionaries
        """
        total_samples = sum(arm['samples'] for arm in arms)
        
        for arm in arms:
            n_i = arm['samples']
            mean = arm['mean_sim']
            var = max(arm['variance'], self.min_variance_threshold)
            
            focus_score = mean
            if total_samples > 1 and n_i > 0:
                # Variance-based confidence term
                focus_score += math.sqrt(max(0.0, 2 * math.log(total_samples) * var / n_i))
                # Sample count penalty
                focus_score += (3 * math.log(total_samples) / n_i)
            
            arm['focus_score'] = focus_score
    
    def _choose_promising_arms(self, arms: List[Dict]) -> List[Dict]:
        """
        Stage 2: Choose promising arms using FOCUS confidence upper-bound scores.
        
        Select the top arms based on FOCUS scores for fine-grained sampling.
        This focuses computational budget on the most promising temporal regions.
        
        Args:
            arms: List of arm dictionaries
            
        Returns:
            List of selected promising arms
        """
        arms_sorted = sorted(arms, key=lambda x: x['focus_score'], reverse=True)
        zoom_count = max(self.min_zoom_segments, int(np.ceil(len(arms) * self.zoom_ratio)))
        zoom_count = min(zoom_count, len(arms))
        
        selected_arms = arms_sorted[:zoom_count]
        return selected_arms
    
    def _fine_sampling_in_arms(
        self,
        selected_arms: List[Dict],
        total_frames: int,
        fine_stride: int,
        region_half_window: int,
        coarse_indices_set: set,
        rng: np.random.Generator
    ) -> List[int]:
        """
        Stage 3: Fine sampling in promising arms with mixed uniform/random strategy.
        
        For each selected arm, densely sample a window around its center using
        a mix of uniform sampling (for coverage) and random sampling (for exploration).
        
        Args:
            selected_arms: List of selected promising arms
            total_frames: Total number of frames in video
            fine_stride: Stride for fine sampling
            region_half_window: Half window size around arm centers (in frames)
            coarse_indices_set: Set of already sampled coarse indices (to avoid duplicates)
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
            uniform_count = int(round(required_samples * self.fine_uniform_ratio))
            random_count = required_samples - uniform_count
            
            # Uniform sampling for coverage
            if uniform_count > 0:
                if window_size <= uniform_count:
                    # If window is small, just take center
                    center_idx = (start + end) // 2
                    if center_idx not in coarse_indices_set and center_idx not in candidate_fine:
                        candidate_fine.add(center_idx)
                else:
                    # Uniform spacing across window
                    interval_size = window_size / uniform_count
                    for i in range(uniform_count):
                        uniform_idx = start + int(i * interval_size + interval_size / 2)
                        uniform_idx = min(uniform_idx, end)
                        if uniform_idx not in coarse_indices_set and uniform_idx not in candidate_fine:
                            candidate_fine.add(uniform_idx)
            
            # Random sampling for exploration
            if random_count > 0:
                available_indices = [i for i in range(start, end + 1)
                                   if i not in coarse_indices_set and i not in candidate_fine]
                if available_indices:
                    actual_random_count = min(random_count, len(available_indices))
                    random_indices = rng.choice(available_indices, size=actual_random_count, replace=False)
                    for idx in random_indices:
                        candidate_fine.add(int(idx))
        
        return sorted(list(candidate_fine))
    
    def _select_top_frames(
        self,
        all_sampled_scores: Dict[int, float],
        k: int
    ) -> List[int]:
        """
        Stage 4a: Select top-ranked frames from all sampled scores.
        
        Directly select the highest scoring frames based on top_ratio.
        This ensures that the best discovered frames are always included.
        
        Args:
            all_sampled_scores: Dictionary mapping frame indices to similarity scores
            k: Total number of keyframes to select
            
        Returns:
            List of selected top frame indices
        """
        if not all_sampled_scores or k <= 0:
            return []
        
        num_computed_frames = len(all_sampled_scores)
        k_top = int(round(self.top_ratio * min(k, num_computed_frames)))
        
        sorted_sampled = sorted(all_sampled_scores.items(), key=lambda x: x[1], reverse=True)
        top_frames = [idx for idx, _ in sorted_sampled[:k_top]]
        
        return top_frames
    
    def _select_remaining_frames(
        self,
        arms: List[Dict],
        remaining_count: int,
        all_sampled_scores: Dict[int, float],
        selected_frames: List[int],
        fps: float,
        min_gap_sec: float,
        rng: np.random.Generator
    ) -> Tuple[List[int], List[float], List[int]]:
        """
        Stage 4b: Select remaining frames using FOCUS arm selection strategy.
        
        Strategy:
        1. Select top S = clamp(zoom_ratio * num_arms, min_arms, max_arms) arms
        2. Evenly distribute remaining_count across these S arms
        3. Within each arm, use interpolation or random sampling based on interpolation_method
        4. Respect minimum temporal gap constraints if specified
        
        Args:
            arms: List of arm dictionaries
            remaining_count: Number of remaining frames to select
            all_sampled_scores: Dictionary of all sampled scores
            selected_frames: List of already selected frames
            fps: Video frame rate
            min_gap_sec: Minimum temporal gap between selections in seconds
            rng: Random number generator
            
        Returns:
            Tuple of (new_frames, arm_selection_probs, arm_selection_counts)
                - new_frames: List of newly selected frame indices
                - arm_selection_probs: Probability of each arm being selected
                - arm_selection_counts: Number of frames selected from each arm
        """
        if remaining_count <= 0 or not arms:
            return [], [], []
        
        gap_frames = int(round(min_gap_sec * fps)) if min_gap_sec > 0 else 0
        selected_set = set(selected_frames)
        
        def respects_gap(cand: int, chosen_set: set) -> bool:
            """Check if candidate respects minimum gap constraint."""
            if gap_frames <= 0:
                return True
            for frame in chosen_set:
                if abs(cand - frame) < gap_frames:
                    return False
            return True
        
        # Compute number of top arms using zoom_ratio with hard bounds
        total_arms = len(arms)
        S = int(np.ceil(total_arms * max(0.0, min(1.0, float(self.zoom_ratio)))))
        S = max(self.final_min_arms, S)
        S = min(S, self.final_max_arms)
        S = min(S, total_arms)
        
        arms_sorted = sorted([(i, arm) for i, arm in enumerate(arms)],
                           key=lambda x: x[1]['focus_score'], reverse=True)
        top_arm_entries = arms_sorted[:S]
        
        # Even allocation across top S arms
        base = remaining_count // S
        rem = remaining_count % S
        per_arm_need = [base + (1 if i < rem else 0) for i in range(S)]
        
        arm_selection_counts = [0 for _ in range(total_arms)]
        new_frames = []
        current_selected_set = selected_set.copy()
        
        for rank, (arm_idx, arm) in enumerate(top_arm_entries):
            needed_count = per_arm_need[rank]
            if needed_count == 0:
                continue
            
            arm_start, arm_end = arm['start'], arm['end']
            all_arm_candidates = list(range(arm_start, arm_end + 1))
            available_candidates = [c for c in all_arm_candidates if c not in current_selected_set]
            if not available_candidates:
                continue
            
            if self.interpolation_method == 'uniform':
                # Random sampling within arm (no interpolation)
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
                    arm_start, arm_end, self.interpolation_method
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
                    
                    # Normalize scores to [0, 1]
                    if scores.max() > scores.min():
                        scores = (scores - scores.min()) / (scores.max() - scores.min())
                    else:
                        scores = np.ones_like(scores)
                    
                    # Apply temperature-based softmax
                    logits = scores / max(1e-12, self.temperature)
                    logits = logits - logits.max()  # numerical stability
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
                    # Fallback: random sampling
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
    
    def _prepare_sampling_details(
        self,
        coarse_indices: List[int],
        coarse_similarities: List[float],
        fine_indices: List[int],
        fine_similarities: List[float],
        arms: List[Dict],
        selected_frames: List[int],
        total_frames: int,
        fps: float,
        budget_used: int,
        arm_selection_probs: List[float],
        arm_selection_counts: List[int]
    ) -> Dict:
        """
        Prepare detailed sampling information for export and analysis.
        
        This creates a comprehensive record of the sampling process including:
        - Coarse and fine sampling statistics
        - Arm information and statistics
        - Final selection results
        - Video metadata
        
        Args:
            coarse_indices: Frame indices sampled in coarse stage
            coarse_similarities: Corresponding similarity scores
            fine_indices: Frame indices sampled in fine stage
            fine_similarities: Corresponding similarity scores
            arms: List of arm dictionaries with statistics
            selected_frames: Final selected frame indices
            total_frames: Total frames in video
            fps: Video frame rate
            budget_used: Total number of similarity computations
            arm_selection_probs: Probability of each arm being selected
            arm_selection_counts: Number of frames selected from each arm
            
        Returns:
            Dictionary containing all sampling details
        """
        
        def create_temporal_order(frame_indices: List[int], similarities: List[float], fps: float) -> List[Dict]:
            """Helper: Create temporal order list with timestamps."""
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

