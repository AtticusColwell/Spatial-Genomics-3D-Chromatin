import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List

@dataclass
class ChromatinContact:
    chromosome: str
    start_pos: int
    end_pos: int
    contact_frequency: float
    resolution: int
    metadata: Optional[Dict] = None

@dataclass 
class TADomain:
    chromosome: str
    start: int
    end: int
    boundary_strength: float
    resolution: int
    
@dataclass
class ChromatinLoop:
    chromosome: str
    anchor1_start: int
    anchor1_end: int
    anchor2_start: int
    anchor2_end: int
    loop_strength: float
    p_value: float

class SpatialGenomicsData:
    def __init__(self):
        self.contact_matrix = None
        self.spatial_coords = None
        self.gene_expression = None
        self.metadata = {}
        self.resolution = None
        
    def load_hic_matrix(self, matrix_path: str):
        pass
        
    def load_spatial_data(self, spatial_path: str):
        pass
        
    def get_contacts_in_region(self, chromosome: str, start: int, end: int):
        pass

class GenomeCoordinates:
    def __init__(self, chromosome: str, start: int, end: int):
        self.chromosome = chromosome
        self.start = start
        self.end = end
        
    def overlaps(self, other: 'GenomeCoordinates') -> bool:
        if self.chromosome != other.chromosome:
            return False
        return not (self.end <= other.start or self.start >= other.end)
        
    def distance_to(self, other: 'GenomeCoordinates') -> int:
        if self.chromosome != other.chromosome:
            return float('inf')
        if self.overlaps(other):
            return 0
        return min(abs(self.start - other.end), abs(self.end - other.start))