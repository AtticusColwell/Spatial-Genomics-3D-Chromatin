import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import h5py
from spatial_chrom3d.io.loaders import (
    load_hic_matrix,
    load_bed_file,
    load_hic_cooler,
    load_spatial_transcriptomics
)
from spatial_chrom3d.core.data_structures import SpatialGenomicsData

class TestDataLoaders:
    
    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        
        self.sample_matrix = np.array([
            [0, 10, 5, 2],
            [10, 0, 15, 8],
            [5, 15, 0, 20],
            [2, 8, 20, 0]
        ])
        
        self.sample_bed_data = [
            ["chr1", 1000, 2000],
            ["chr1", 5000, 7000],
            ["chr2", 10000, 15000]
        ]

    def test_load_text_matrix(self):
        matrix_file = self.temp_dir / "test_matrix.txt"
        np.savetxt(matrix_file, self.sample_matrix, delimiter='\t')
        
        loaded = load_hic_matrix(str(matrix_file))
        np.testing.assert_array_equal(loaded, self.sample_matrix)

    def test_load_tsv_matrix(self):
        matrix_file = self.temp_dir / "test_matrix.tsv"
        np.savetxt(matrix_file, self.sample_matrix, delimiter='\t')
        
        loaded = load_hic_matrix(str(matrix_file))
        np.testing.assert_array_equal(loaded, self.sample_matrix)

    def test_load_matrix_auto_format(self):
        matrix_file = self.temp_dir / "test_matrix.txt"
        np.savetxt(matrix_file, self.sample_matrix, delimiter='\t')
        
        loaded = load_hic_matrix(str(matrix_file), format='auto')
        np.testing.assert_array_equal(loaded, self.sample_matrix)

    def test_load_unsupported_format(self):
        with pytest.raises(ValueError, match="Unsupported format"):
            load_hic_matrix("fake_file.xyz")

    def test_load_bed_file(self):
        bed_file = self.temp_dir / "test.bed"
        with open(bed_file, 'w') as f:
            for row in self.sample_bed_data:
                f.write('\t'.join(map(str, row)) + '\n')
        
        df = load_bed_file(str(bed_file))
        
        assert df.shape == (3, 3)
        assert list(df.columns) == ['chromosome', 'start', 'end']
        assert df.iloc[0]['chromosome'] == 'chr1'
        assert df.iloc[0]['start'] == 1000
        assert df.iloc[0]['end'] == 2000

    def test_load_bed_file_extra_columns(self):
        bed_file = self.temp_dir / "test_extra.bed"
        with open(bed_file, 'w') as f:
            f.write("chr1\t1000\t2000\textra_col\n")
            f.write("chr2\t5000\t7000\tanother_col\n")
        
        df = load_bed_file(str(bed_file))
        
        assert df.shape == (2, 3)
        assert list(df.columns) == ['chromosome', 'start', 'end']

    def test_load_cooler_missing_import(self):
        with pytest.raises(ImportError, match="cooler package required"):
            load_hic_cooler("fake_file.cool")

    def test_load_spatial_missing_import(self):
        with pytest.raises(ImportError, match="scanpy required"):
            load_spatial_transcriptomics("fake_file.h5ad")

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_hic_matrix("nonexistent_file.txt")

class TestFileFormats:
    
    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def test_matrix_with_headers(self):
        matrix_file = self.temp_dir / "with_headers.txt"
        data = np.array([[1, 2], [3, 4]])
        
        with open(matrix_file, 'w') as f:
            f.write("col1\tcol2\n")
            f.write("1\t2\n")
            f.write("3\t4\n")
        
        try:
            loaded = load_hic_matrix(str(matrix_file))
        except ValueError:
            pass

    def test_irregular_matrix(self):
        matrix_file = self.temp_dir / "irregular.txt"
        
        with open(matrix_file, 'w') as f:
            f.write("1\t2\t3\n")
            f.write("4\t5\n")
            f.write("6\t7\t8\t9\n")
        
        try:
            loaded = load_hic_matrix(str(matrix_file))
        except (ValueError, pd.errors.ParserError):
            pass

    def test_empty_file(self):
        empty_file = self.temp_dir / "empty.txt"
        empty_file.touch()
        
        with pytest.raises((ValueError, pd.errors.EmptyDataError)):
            load_hic_matrix(str(empty_file))