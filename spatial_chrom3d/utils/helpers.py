import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional

def genomic_distance(pos1: int, pos2: int, chromosome1: str = None, chromosome2: str = None) -> int:
    if chromosome1 and chromosome2 and chromosome1 != chromosome2:
        return float('inf')
    return abs(pos1 - pos2)

def bin_coordinates(start: int, end: int, resolution: int) -> Tuple[int, int]:
    start_bin = start // resolution
    end_bin = end // resolution
    return start_bin, end_bin

def coordinates_to_bins(coordinates: List[Tuple[int, int]], resolution: int) -> List[Tuple[int, int]]:
    return [bin_coordinates(start, end, resolution) for start, end in coordinates]

def create_genomic_bins(chromosome_sizes: Dict[str, int], resolution: int) -> pd.DataFrame:
    bins = []
    for chrom, size in chromosome_sizes.items():
        for start in range(0, size, resolution):
            end = min(start + resolution, size)
            bins.append({
                'chromosome': chrom,
                'start': start,
                'end': end,
                'bin_id': len(bins)
            })
    return pd.DataFrame(bins)

def calculate_matrix_stats(matrix: np.ndarray) -> Dict[str, float]:
    return {
        'total_contacts': np.sum(matrix),
        'mean_contact': np.mean(matrix),
        'std_contact': np.std(matrix),
        'max_contact': np.max(matrix),
        'sparsity': np.sum(matrix == 0) / matrix.size,
        'non_zero_contacts': np.sum(matrix > 0)
    }

def smooth_track(values: np.ndarray, window_size: int = 3) -> np.ndarray:
    return np.convolve(values, np.ones(window_size)/window_size, mode='same')

def percentile_normalize(matrix: np.ndarray, lower: float = 1, upper: float = 99) -> np.ndarray:
    lower_val = np.percentile(matrix, lower)
    upper_val = np.percentile(matrix, upper)
    clipped = np.clip(matrix, lower_val, upper_val)
    return (clipped - lower_val) / (upper_val - lower_val)

def merge_overlapping_regions(regions: List[Tuple[int, int]], min_gap: int = 0) -> List[Tuple[int, int]]:
    if not regions:
        return []
    
    sorted_regions = sorted(regions, key=lambda x: x[0])
    merged = [sorted_regions[0]]
    
    for current in sorted_regions[1:]:
        last = merged[-1]
        if current[0] <= last[1] + min_gap:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    
    return merged