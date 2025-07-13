import numpy as np
from typing import Union, List, Tuple, Dict, Any
from pathlib import Path

def validate_matrix_format(matrix: np.ndarray) -> bool:
    if not isinstance(matrix, np.ndarray):
        return False
    if matrix.ndim != 2:
        return False
    if matrix.shape[0] != matrix.shape[1]:
        return False
    return True

def validate_genomic_coordinates(chromosome: str, start: int, end: int) -> bool:
    if not isinstance(chromosome, str):
        return False
    if not isinstance(start, int) or not isinstance(end, int):
        return False
    if start < 0 or end < 0:
        return False
    if start >= end:
        return False
    return True

def validate_resolution(resolution: int) -> bool:
    return isinstance(resolution, int) and resolution > 0

def validate_file_exists(file_path: Union[str, Path]) -> bool:
    return Path(file_path).exists()

def validate_hic_matrix(matrix: np.ndarray, symmetric: bool = True) -> Tuple[bool, List[str]]:
    errors = []
    
    if not validate_matrix_format(matrix):
        errors.append("Matrix must be 2D and square")
        return False, errors
    
    if symmetric and not np.allclose(matrix, matrix.T, rtol=1e-5):
        errors.append("Matrix should be symmetric for Hi-C data")
    
    if np.any(matrix < 0):
        errors.append("Matrix contains negative values")
    
    if np.all(matrix == 0):
        errors.append("Matrix is empty (all zeros)")
    
    return len(errors) == 0, errors

def validate_tad_list(tads: List[Any]) -> Tuple[bool, List[str]]:
    errors = []
    
    for i, tad in enumerate(tads):
        if not hasattr(tad, 'start') or not hasattr(tad, 'end'):
            errors.append(f"TAD {i} missing start or end attribute")
        elif tad.start >= tad.end:
            errors.append(f"TAD {i} has invalid coordinates: start >= end")
    
    return len(errors) == 0, errors