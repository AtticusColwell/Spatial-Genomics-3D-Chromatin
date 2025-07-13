import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from ..core.data_structures import TADomain, ChromatinLoop

def tads_to_dataframe(tads: List[TADomain]) -> pd.DataFrame:
    data = []
    for tad in tads:
        data.append({
            'chromosome': tad.chromosome,
            'start': tad.start,
            'end': tad.end,
            'boundary_strength': tad.boundary_strength,
            'resolution': tad.resolution,
            'size': tad.end - tad.start
        })
    return pd.DataFrame(data)

def loops_to_dataframe(loops: List[ChromatinLoop]) -> pd.DataFrame:
    data = []
    for loop in loops:
        data.append({
            'chromosome': loop.chromosome,
            'anchor1_start': loop.anchor1_start,
            'anchor1_end': loop.anchor1_end,
            'anchor2_start': loop.anchor2_start,
            'anchor2_end': loop.anchor2_end,
            'loop_strength': loop.loop_strength,
            'p_value': loop.p_value,
            'distance': loop.anchor2_start - loop.anchor1_end
        })
    return pd.DataFrame(data)

def matrix_to_sparse_format(matrix: np.ndarray, threshold: float = 0) -> Dict[str, np.ndarray]:
    rows, cols = np.where(matrix > threshold)
    values = matrix[rows, cols]
    
    return {
        'rows': rows,
        'cols': cols,
        'values': values,
        'shape': matrix.shape
    }

def sparse_to_matrix(sparse_data: Dict[str, np.ndarray]) -> np.ndarray:
    matrix = np.zeros(sparse_data['shape'])
    matrix[sparse_data['rows'], sparse_data['cols']] = sparse_data['values']
    return matrix

def coordinates_to_bed_format(coordinates: List[Tuple[str, int, int]], 
                             output_file: str = None) -> pd.DataFrame:
    df = pd.DataFrame(coordinates, columns=['chromosome', 'start', 'end'])
    if output_file:
        df.to_csv(output_file, sep='\t', header=False, index=False)
    return df