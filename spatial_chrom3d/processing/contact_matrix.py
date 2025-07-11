import numpy as np
from scipy import ndimage
from typing import Tuple

def normalize_contact_matrix(matrix: np.ndarray, method: str = 'ice') -> np.ndarray:
    if method == 'ice':
        return apply_ice_normalization(matrix)
    elif method == 'kr':
        return apply_kr_normalization(matrix)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def apply_ice_normalization(matrix: np.ndarray, max_iter: int = 100, tol: float = 1e-5) -> np.ndarray:
    matrix = matrix.astype(float)
    matrix[matrix == 0] = np.nan
    
    for iteration in range(max_iter):
        row_sums = np.nansum(matrix, axis=1)
        col_sums = np.nansum(matrix, axis=0)
        
        row_mean = np.nanmean(row_sums)
        col_mean = np.nanmean(col_sums)
        
        row_factors = np.sqrt(row_mean / row_sums)
        col_factors = np.sqrt(col_mean / col_sums)
        
        row_factors[np.isnan(row_factors)] = 0
        col_factors[np.isnan(col_factors)] = 0
        
        matrix = matrix * row_factors[:, np.newaxis]
        matrix = matrix * col_factors[np.newaxis, :]
        
        if iteration > 0 and np.allclose(row_factors, 1, rtol=tol):
            break
            
    matrix[np.isnan(matrix)] = 0
    return matrix

def apply_kr_normalization(matrix: np.ndarray) -> np.ndarray:
    return matrix

def remove_diagonal(matrix: np.ndarray, diag_size: int = 2) -> np.ndarray:
    result = matrix.copy()
    for i in range(-diag_size, diag_size + 1):
        if i == 0:
            continue
        np.fill_diagonal(result[max(0, i):, max(0, -i):], 0)
    np.fill_diagonal(result, 0)
    return result

def smooth_matrix(matrix: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return ndimage.gaussian_filter(matrix, sigma=sigma)

def compute_observed_expected(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    oe_matrix = np.zeros_like(matrix)
    
    for d in range(n):
        if d == 0:
            continue
            
        diagonal_values = []
        for i in range(n - d):
            if matrix[i, i + d] > 0:
                diagonal_values.append(matrix[i, i + d])
                
        if diagonal_values:
            expected = np.mean(diagonal_values)
            for i in range(n - d):
                if matrix[i, i + d] > 0:
                    oe_matrix[i, i + d] = matrix[i, i + d] / expected
                    oe_matrix[i + d, i] = oe_matrix[i, i + d]
                    
    return oe_matrix