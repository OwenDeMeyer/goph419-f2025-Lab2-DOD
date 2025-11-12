import numpy as np
import pytest
from linalg_interp.py import spline_function

def test_linear_exact():
    xd = np.linspace(0, 5, 6)
    yd = 3 * xd + 2
    f = spline_function(xd, yd, order=1)
    x_test = np.linspace(0, 5, 20)
    assert np.allclose(f(x_test), 3 * x_test + 2)

def test_cubic_exact_for_cubic_data():
    xd = np.linspace(0, 4, 6)
    yd = xd**3 - 2 * xd**2 + xd + 5
    f = spline_function(xd, yd, order=3)
    x_test = np.linspace(0, 4, 10)
    assert np.allclose(f(x_test), xd**3 - 2 * xd**2 + xd + 5, atol=1e-8)

def test_out_of_bounds():
    xd = np.array([0, 1, 2])
    yd = np.array([1, 2, 3])
    f = spline_function(xd, yd)
    with pytest.raises(ValueError):
        f(3.5)

