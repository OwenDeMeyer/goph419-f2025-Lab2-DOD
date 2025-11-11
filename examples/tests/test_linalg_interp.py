import sys
import os
# Add the src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
import numpy as np
import pytest
from linalg_interp import gauss_iter_solve

def test_single_rhs_gauss_seidel():
    A = np.array([[4, 1, 2],
                  [3, 5, 1],
                  [1, 1, 3]], dtype=float)
    b = np.array([4, 7, 3], dtype=float)
    x_ref = np.linalg.solve(A, b)

    x = gauss_iter_solve(A, b, x0, tol=1e-8, alg='seidel')
    assert np.allclose(x, x_ref, atol=1e-8)

def test_single_rhs_jacobi():
    A = np.array([[10, 1, 1],
                  [2, 10, 1],
                  [2, 2, 10]], dtype=float)
    b = np.array([12, 13, 14], dtype=float)
    x_ref = np.linalg.solve(A, b)

    x = gauss_iter_solve(A, b, x0, tol=1e-8, alg='jacobi')
    assert np.allclose(x, x_ref, atol=1e-8)

def test_inverse_computation():
    A = np.array([[4, 2],
                  [1, 3]], dtype=float)
    I = np.eye(2)
    A_inv = gauss_iter_solve(A, I, x0, tol=1e-8, alg='seidel')
    assert np.allclose(A @ A_inv, np.eye(2), atol=1e-8)

def test_invalid_alg():
    A = np.eye(2)
    b = np.ones(2)
    with pytest.raises(ValueError):
        gauss_iter_solve(A, b, x0, tol=1e-8, alg='invalid')

def test_dimension_mismatch():
    A = np.eye(3)
    b = np.ones(2)
    with pytest.raises(ValueError):
        gauss_iter_solve(A, b, x0, tol=1e-8, alg='seidel')

