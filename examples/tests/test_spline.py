# To run tests, while in the main directory, run -> pytest examples/tests/test_spline.py -v and it shall show all the tests passing, make sure scipy and other packages are downloaded first.
import sys
import os
# Add the src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import numpy as np
import pytest
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
    assert not np.allclose(f(x_test), xd**2 + 2*xd + 1)

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
    assert np.allclose(f(x_test), us(x_test), atol=1e-8)

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
    assert np.allclose(f(x_test), us(x_test), atol=1e-8)


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

