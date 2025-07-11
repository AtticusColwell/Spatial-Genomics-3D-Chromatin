import numpy as np
from scipy import signal
from typing import List, Tuple
from ..core.data_structures import TADomain, ChromatinLoop

def detect_tads(contact_matrix: np.ndarray, resolution: int = 10000, 
                min_size: int = 3, max_size: int = 100) -> List[TADomain]:
    
    insulation_scores = compute_insulation_score(contact_matrix)
    boundaries = find_insulation_minima(insulation_scores)
    
    tads = []
    for i in range(len(boundaries) - 1):
        start_bin = boundaries[i]
        end_bin = boundaries[i + 1]
        
        if min_size <= (end_bin - start_bin) <= max_size:
            boundary_strength = insulation_scores[start_bin] + insulation_scores[end_bin]
            
            tad = TADomain(
                chromosome='chr1',
                start=start_bin * resolution,
                end=end_bin * resolution,
                boundary_strength=boundary_strength,
                resolution=resolution
            )
            tads.append(tad)
            
    return tads

def compute_insulation_score(matrix: np.ndarray, window_size: int = 5) -> np.ndarray:
    n = matrix.shape[0]
    insulation_scores = np.zeros(n)
    
    for i in range(window_size, n - window_size):
        upstream = matrix[i-window_size:i, i-window_size:i]
        downstream = matrix[i:i+window_size, i:i+window_size]
        
        upstream_sum = np.sum(upstream)
        downstream_sum = np.sum(downstream)
        cross_sum = np.sum(matrix[i-window_size:i, i:i+window_size])
        
        if upstream_sum + downstream_sum > 0:
            insulation_scores[i] = cross_sum / (upstream_sum + downstream_sum)
            
    return insulation_scores

def find_insulation_minima(insulation_scores: np.ndarray, 
                          prominence: float = 0.1) -> List[int]:
    minima, _ = signal.find_peaks(-insulation_scores, prominence=prominence)
    return minima.tolist()

def detect_loops(contact_matrix: np.ndarray, resolution: int = 10000,
                min_distance: int = 10, peak_threshold: float = 2.0) -> List[ChromatinLoop]:
    
    from ..processing.contact_matrix import compute_observed_expected
    oe_matrix = compute_observed_expected(contact_matrix)
    peaks = find_loop_peaks(oe_matrix, peak_threshold)
    
    loops = []
    for peak in peaks:
        i, j = peak
        if abs(i - j) >= min_distance:
            loop_strength = oe_matrix[i, j]
            
            loop = ChromatinLoop(
                chromosome='chr1',
                anchor1_start=i * resolution,
                anchor1_end=(i + 1) * resolution,
                anchor2_start=j * resolution,
                anchor2_end=(j + 1) * resolution,
                loop_strength=loop_strength,
                p_value=0.001
            )
            loops.append(loop)
            
    return loops

def find_loop_peaks(oe_matrix: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
    from scipy import ndimage
    
    peaks = []
    filtered_matrix = ndimage.maximum_filter(oe_matrix, size=3)
    
    peak_mask = (oe_matrix == filtered_matrix) & (oe_matrix > threshold)
    peak_coords = np.where(peak_mask)
    
    for i, j in zip(peak_coords[0], peak_coords[1]):
        if i < j:
            peaks.append((i, j))
            
    return peaks