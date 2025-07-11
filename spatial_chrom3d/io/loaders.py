import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Optional, Union, Dict, Any
from ..core.data_structures import SpatialGenomicsData

def load_hic_cooler(file_path: str, resolution: Optional[int] = None) -> SpatialGenomicsData:
    try:
        import cooler
        c = cooler.Cooler(file_path)
        
        data = SpatialGenomicsData()
        data.contact_matrix = c.matrix(balance=True)[:]
        data.resolution = c.binsize
        data.metadata = {
            'chromosomes': c.chromnames,
            'bins': c.bins()[:],
            'format': 'cooler'
        }
        return data
    except ImportError:
        raise ImportError("cooler package required for .cool files")

def load_hic_matrix(file_path: str, format: str = 'auto') -> np.ndarray:
    if format == 'auto':
        format = Path(file_path).suffix.lower()
    
    if format == '.cool':
        return load_hic_cooler(file_path)
    elif format in ['.txt', '.tsv']:
        return pd.read_csv(file_path, sep='\t', header=None).values
    else:
        raise ValueError(f"Unsupported format: {format}")

def load_spatial_transcriptomics(file_path: str, format: str = 'h5ad') -> Dict[str, Any]:
    try:
        import scanpy as sc
        
        if format == 'h5ad':
            adata = sc.read_h5ad(file_path)
            return {
                'expression': adata.X,
                'coordinates': adata.obsm.get('spatial', None),
                'gene_names': adata.var_names,
                'cell_names': adata.obs_names,
                'metadata': adata.obs
            }
        else:
            raise ValueError(f"Unsupported spatial format: {format}")
    except ImportError:
        raise ImportError("scanpy required for spatial transcriptomics data")

def load_bed_file(file_path: str) -> pd.DataFrame:
    columns = ['chromosome', 'start', 'end']
    df = pd.read_csv(file_path, sep='\t', header=None, usecols=[0, 1, 2])
    df.columns = columns
    return df