import pytest
import numpy as np
from spatial_chrom3d.processing.contact_matrix import (
    normalize_contact_matrix,
    apply_ice_normalization,
    apply_kr_normalization,
    remove_diagonal,
    smooth_matrix,
    compute_observed_expected
)

class TestContactMatrixProcessing:
    
    def setup_method(self):
        self.test_matrix = np.array([
            [0, 10, 5, 2, 1],
            [10, 0, 15, 8, 3],
            [5, 15, 0, 20, 10],
            [2, 8, 20, 0, 25],
            [1, 3, 10, 25, 0]
        ], dtype=float)
        
        self.asymmetric_matrix = np.array([
            [0, 10, 5],
            [12, 0, 15],
            [6, 18, 0]
        ], dtype=float)
        
        self.zero_matrix = np.zeros((5, 5))
        
        self.sparse_matrix = np.array([
            [0, 1, 0, 0, 0],
            [1, 0, 2, 0, 0],
            [0, 2, 0, 3, 0],
            [0, 0, 3, 0, 4],
            [0, 0, 0, 4, 0]
        ], dtype=float)

    def test_ice_normalization_basic(self):
        normalized = normalize_contact_matrix(self.test_matrix, method='ice')
        
        assert normalized.shape == self.test_matrix.shape
        assert not np.array_equal(normalized, self.test_matrix)
        assert np.all(normalized >= 0)
        np.testing.assert_allclose(normalized, normalized.T, rtol=1e-10)

    def test_invalid_normalization_method(self):
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_contact_matrix(self.test_matrix, method='nonexistent')

    def test_ice_implementation(self):
        normalized = apply_ice_normalization(self.test_matrix.copy())
        
        assert normalized.shape == self.test_matrix.shape
        np.testing.assert_allclose(normalized, normalized.T, rtol=1e-10)
        assert not np.any(np.isnan(normalized))
        
        row_sums = np.sum(normalized, axis=1)
        col_sums = np.sum(normalized, axis=0)
        np.testing.assert_allclose(row_sums, col_sums, rtol=1e-5)

    def test_ice_with_zeros(self):
        normalized = apply_ice_normalization(self.zero_matrix.copy())
        np.testing.assert_array_equal(normalized, self.zero_matrix)

    def test_ice_convergence_parameters(self):
        normalized = apply_ice_normalization(self.test_matrix.copy(), max_iter=5, tol=1e-3)
        
        assert normalized.shape == self.test_matrix.shape
        assert not np.any(np.isnan(normalized))

    def test_kr_normalization(self):
        normalized = apply_kr_normalization(self.test_matrix.copy())
        np.testing.assert_array_equal(normalized, self.test_matrix)

    def test_diagonal_removal(self):
        no_diag = remove_diagonal(self.test_matrix.copy())
        
        assert np.all(np.diag(no_diag) == 0)
        
        mask = ~np.eye(self.test_matrix.shape[0], dtype=bool)
        np.testing.assert_array_equal(no_diag[mask], self.test_matrix[mask])

    def test_diagonal_removal_custom_size(self):
        diag_size = 1
        no_diag = remove_diagonal(self.test_matrix.copy(), diag_size=diag_size)
        
        assert np.all(np.diag(no_diag) == 0)
        
        for i in range(-diag_size, diag_size + 1):
            if i != 0:
                diag_vals = np.diag(no_diag, k=i)
                assert np.all(diag_vals == 0)

    def test_matrix_smoothing(self):
        smoothed = smooth_matrix(self.test_matrix.copy())
        
        assert smoothed.shape == self.test_matrix.shape
        assert not np.array_equal(smoothed, self.test_matrix)
        
        sum_diff = abs(np.sum(smoothed) - np.sum(self.test_matrix))
        assert sum_diff < np.sum(self.test_matrix) * 0.1

    def test_smoothing_sigma_effects(self):
        smoothed_small = smooth_matrix(self.test_matrix.copy(), sigma=0.5)
        smoothed_large = smooth_matrix(self.test_matrix.copy(), sigma=2.0)
        
        diff_small = np.sum(np.abs(smoothed_small - self.test_matrix))
        diff_large = np.sum(np.abs(smoothed_large - self.test_matrix))
        
        assert diff_large > diff_small

    def test_observed_expected_calculation(self):
        oe_matrix = compute_observed_expected(self.test_matrix)
        
        assert oe_matrix.shape == self.test_matrix.shape
        np.testing.assert_allclose(oe_matrix, oe_matrix.T, rtol=1e-10)
        assert np.all(oe_matrix >= 0)
        assert np.all(np.diag(oe_matrix) == 0)

    def test_observed_expected_with_zeros(self):
        matrix_with_zeros = self.test_matrix.copy()
        matrix_with_zeros[0, 1] = 0
        matrix_with_zeros[1, 0] = 0
        
        oe_matrix = compute_observed_expected(matrix_with_zeros)
        
        assert oe_matrix.shape == matrix_with_zeros.shape
        assert oe_matrix[0, 1] == 0
        assert oe_matrix[1, 0] == 0

    def test_observed_expected_sparse(self):
        oe_matrix = compute_observed_expected(self.sparse_matrix)
        
        assert oe_matrix.shape == self.sparse_matrix.shape
        assert np.all(oe_matrix >= 0)

class TestMatrixDimensions:
    
    def test_small_matrices(self):
        small_matrix = np.array([[1, 2], [2, 1]], dtype=float)
        
        normalized = normalize_contact_matrix(small_matrix, method='ice')
        assert normalized.shape == (2, 2)
        
        no_diag = remove_diagonal(small_matrix)
        assert no_diag.shape == (2, 2)
        
        smoothed = smooth_matrix(small_matrix)
        assert smoothed.shape == (2, 2)

    def test_empty_matrices(self):
        empty_matrix = np.array([]).reshape(0, 0)
        
        with pytest.raises((ValueError, IndexError)):
            normalize_contact_matrix(empty_matrix)

    def test_single_element(self):
        single_matrix = np.array([[5.0]])
        
        normalized = apply_ice_normalization(single_matrix)
        assert normalized.shape == (1, 1)

class TestEdgeCases:
    
    def test_infinite_values(self):
        inf_matrix = np.array([
            [0, np.inf, 1],
            [np.inf, 0, 2],
            [1, 2, 0]
        ], dtype=float)
        
        try:
            normalized = apply_ice_normalization(inf_matrix)
            assert normalized.shape == inf_matrix.shape
        except (ValueError, RuntimeWarning):
            pass

    def test_nan_values(self):
        nan_matrix = np.array([
            [0, np.nan, 1],
            [np.nan, 0, 2],
            [1, 2, 0]
        ], dtype=float)
        
        normalized = apply_ice_normalization(nan_matrix)
        assert normalized.shape == nan_matrix.shape
        assert not np.any(np.isnan(normalized))

    def test_large_values(self):
        large_matrix = np.array([
            [0, 1e10, 1e8],
            [1e10, 0, 1e9],
            [1e8, 1e9, 0]
        ], dtype=float)
        
        normalized = apply_ice_normalization(large_matrix)
        assert normalized.shape == large_matrix.shape
        assert np.all(np.isfinite(normalized))

    def test_precision_issues(self):
        precision_matrix = np.array([
            [0, 1e-15, 1e-10],
            [1e-15, 0, 1e-12],
            [1e-10, 1e-12, 0]
        ], dtype=float)
        
        normalized = apply_ice_normalization(precision_matrix)
        assert normalized.shape == precision_matrix.shape
        assert not np.any(np.isnan(normalized))

class TestParameterRanges:
    
    def setup_method(self):
        self.matrix = np.array([
            [0, 10, 5],
            [10, 0, 15],
            [5, 15, 0]
        ], dtype=float)

    def test_ice_iterations(self):
        norm_few = apply_ice_normalization(self.matrix.copy(), max_iter=1)
        norm_many = apply_ice_normalization(self.matrix.copy(), max_iter=100)
        
        assert norm_few.shape == self.matrix.shape
        assert norm_many.shape == self.matrix.shape
        assert not np.any(np.isnan(norm_few))
        assert not np.any(np.isnan(norm_many))

    def test_ice_tolerance(self):
        norm_strict = apply_ice_normalization(self.matrix.copy(), tol=1e-10)
        norm_loose = apply_ice_normalization(self.matrix.copy(), tol=1e-2)
        
        assert norm_strict.shape == self.matrix.shape
        assert norm_loose.shape == self.matrix.shape

    def test_smoothing_ranges(self):
        smooth_none = smooth_matrix(self.matrix, sigma=0)
        smooth_heavy = smooth_matrix(self.matrix, sigma=5.0)
        
        assert smooth_none.shape == self.matrix.shape
        assert smooth_heavy.shape == self.matrix.shape
        
        diff_none = np.sum(np.abs(smooth_none - self.matrix))
        diff_heavy = np.sum(np.abs(smooth_heavy - self.matrix))
        assert diff_none < diff_heavy

    def test_diagonal_sizes(self):
        no_diag_0 = remove_diagonal(self.matrix.copy(), diag_size=0)
        assert np.all(np.diag(no_diag_0) == 0)
        
        no_diag_large = remove_diagonal(self.matrix.copy(), diag_size=10)
        assert np.all(np.diag(no_diag_large) == 0)