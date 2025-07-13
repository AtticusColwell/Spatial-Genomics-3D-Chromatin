import pytest
import numpy as np
from spatial_chrom3d.core.data_structures import TADomain, ChromatinLoop, GenomeCoordinates

def test_genome_coordinates_overlap():
    coord1 = GenomeCoordinates("chr1", 1000, 2000)
    coord2 = GenomeCoordinates("chr1", 1500, 2500)
    coord3 = GenomeCoordinates("chr1", 3000, 4000)
    
    assert coord1.overlaps(coord2)
    assert not coord1.overlaps(coord3)

def test_genome_coordinates_distance():
    coord1 = GenomeCoordinates("chr1", 1000, 2000)
    coord2 = GenomeCoordinates("chr1", 3000, 4000)
    coord3 = GenomeCoordinates("chr2", 1000, 2000)
    
    assert coord1.distance_to(coord2) == 1000
    assert coord1.distance_to(coord3) == float('inf')

def test_tad_creation():
    tad = TADomain(
        chromosome="chr1",
        start=100000,
        end=200000,
        boundary_strength=0.8,
        resolution=10000
    )
    
    assert tad.chromosome == "chr1"
    assert tad.start == 100000
    assert tad.end == 200000
    assert tad.boundary_strength == 0.8

def test_loop_creation():
    loop = ChromatinLoop(
        chromosome="chr1",
        anchor1_start=100000,
        anchor1_end=110000,
        anchor2_start=200000,
        anchor2_end=210000,
        loop_strength=2.5,
        p_value=0.001
    )
    
    assert loop.chromosome == "chr1"
    assert loop.loop_strength == 2.5