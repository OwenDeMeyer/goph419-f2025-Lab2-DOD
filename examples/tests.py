# To run tests, while in the main directory, run -> pytest examples/tests/test_linalg_interp.py -v and it shall show all the tests passing. 
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
                  [1, 1, 3]], dtype=float) #Set matrix A
    b = np.array([4, 7, 3], dtype=float) #Set matrix B (vector)
    x_ref = np.linalg.solve(A, b) #Create reference matrix using proven code

    x = gauss_iter_solve(A, b, x0=None, tol=1e-8, alg='seidel') #Run our code 
    assert np.allclose(x, x_ref, atol=1e-8) #Check if our result equals to proven method

def test_single_rhs_jacobi(): #same as first one
    A = np.array([[10, 1, 1],
                  [2, 10, 1],
                  [2, 2, 10]], dtype=float)
    b = np.array([12, 13, 14], dtype=float)
    x_ref = np.linalg.solve(A, b)

    x = gauss_iter_solve(A, b, x0=None, tol=1e-8, alg='jacobi')
    assert np.allclose(x, x_ref, atol=1e-8)

def test_inverse_computation():
    A = np.array([[4, 2],
                  [1, 3]], dtype=float) #Set example A
    I = np.eye(2) #Create Identity matrix
    A_inv = gauss_iter_solve(A, I, x0=None, tol=1e-8, alg='seidel') # Obtain A inverse by multiplying A by the identity matrix
    assert np.allclose(A @ A_inv, np.eye(2), atol=1e-8) #Check if A multiply by A inverse is equal to Identity Matrix

def test_invalid_alg():
    A = np.eye(2) #identity matrix for ex
    b = np.ones(2) # Set example matrix (dont matter)
    with pytest.raises(ValueError): 
        gauss_iter_solve(A, b, x0=None, tol=1e-8, alg='GoKingsGo') #If our function raises an error, test passes

def test_dimension_mismatch():
    A = np.eye(3) #A and B are ex matrices that dont matter for this test
    b = np.ones(2)
    with pytest.raises(ValueError):
        gauss_iter_solve(A, b, x0=None, tol=1e-8, alg='seidel') #will raise an error the test passes. 

def main(): 
    test_single_rhs_gauss_seidel()
    print("test_single_rhs_gauss_seidel: PASS")

    test_single_rhs_jacobi()
    print("test_single_rhs_jacobi: PASS")

    test_inverse_computation()
    print("test_inverse_computation: PASS")

    test_invalid_alg()
    print("test_invalid_alg: PASS")

    test_dimension_mismatch()
    print("test_dimension_mismatch: PASS")

    print("\nAll manual tests passed.")
    

if __name__ == "__main__":
    main()
