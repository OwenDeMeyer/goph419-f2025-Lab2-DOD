# To run tests, while in the main directory, run -> pytest examples/tests.py -v and it shall show all the tests passing. 
import sys
import os
# Add the src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
import numpy as np
import pytest
from linalg_interp import gauss_iter_solve
from linalg_interp import spline_function
from scipy.interpolate import UnivariateSpline

def test_linear_exact():
    # Generate linear data
    xd = np.linspace(0, 5, 6)
    yd = 3 * xd + 2
    # Create a linear spline (order=1)
    f = spline_function(xd, yd, order=1)
    # Test points (finer grid)
    x_test = np.linspace(0, 5, 20)
    # The spline should exactly match the linear function
    assert np.allclose(f(x_test), 3 * x_test + 2)

def test_quadratic_with_linear_spline():
    # Generate quadratic data
    xd = np.linspace(0, 4, 5)
    yd = xd**2 + 2*xd + 1
    # Use a linear spline (order=1) -- should NOT exactly match quadratic
    f = spline_function(xd, yd, order=1)
    x_test = np.linspace(0, 4, 20)
    # Linear spline will approximate, but not exactly recover
    assert not np.allclose(f(x_test), x_test**2 + 2*x_test + 1)

def test_cubic_with_cubic_spline():
    # Generate cubic data
    xd = np.linspace(0, 4, 6)
    yd = xd**3 - 2*xd**2 + xd + 5
    # Cubic spline (order=3)
    f = spline_function(xd, yd, order=3)
    x_test = xd  # Check exact match at original points
    # Cubic spline should exactly recover cubic values at original xd
    assert np.allclose(f(x_test), yd, atol=1e-8)


def test_cubic_spline_vs_univariate_spline():
    # Higher-order polynomial
    xd = np.linspace(0, 2, 10)
    yd = xd**4 - 2*xd**3 + xd**2 - xd + 1
    # Our cubic spline (order=3)
    f = spline_function(xd, yd, order=3)
    # UnivariateSpline with k=3 (cubic), s=0 (interpolation), ext='raise' (no extrapolation)
    us = UnivariateSpline(xd, yd, k=3, s=0, ext='raise')
    # Test points (fine grid)
    x_test = np.linspace(xd[0], xd[-1], 50)
    # Compare outputs
    assert np.allclose(f(x_test), us(x_test), atol=1e-2)

def test_exponential_vs_univariate_spline():
    # Exponential data
    xd = np.linspace(0, 2, 6)
    yd = np.exp(xd)
    # Our cubic spline
    f = spline_function(xd, yd, order=3)
    # UnivariateSpline cubic
    us = UnivariateSpline(xd, yd, k=3, s=0, ext='raise')
    # Test points
    x_test = np.linspace(xd[0], xd[-1], 30)
    # Compare outputs
    assert np.allclose(f(x_test), us(x_test), atol=1e-2)


def test_out_of_bounds():
    xd = np.linspace(0, 5, 6)
    yd = 3 * xd + 2
    f = spline_function(xd, yd, order=1)
    # x < xmin should raise ValueError
    with pytest.raises(ValueError):
        f(-1)
    # x > xmax should raise ValueError
    with pytest.raises(ValueError):
        f(10)
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

    test_linear_exact()

    test_quadratic_with_linear_spline()

    test_cubic_with_cubic_spline()

    test_cubic_spline_vs_univariate_spline()

    test_exponential_vs_univariate_spline()

    test_out_of_bounds()
    
    print("\nAll manual tests passed.")
    

if __name__ == "__main__":
    main()
