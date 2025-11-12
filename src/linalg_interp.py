import numpy as np
import warnings
from numpy.linalg import solve
def gauss_iter_solve(A, b, x0 = None, tol = 1e-8, alg = 'seidel'):
    """
    Solve the linear system A x = b using the Gauss-Seidel or Jacobi iterative method
    A : array_like: Coefficient matrix (must be square)
    b : array_like: Right-hand-side vector or matrix
    x0 : array_like, optional: Initial guess for the solution. Default is None (uses zeros)
    tol : float, optional: Relative error tolerance for convergence. Default is 1e-8
    alg : str, optional: Iteration algorithm: 'seidel' (default) or 'jacobi'. Case-insensitive.

    Returns
    x : numpy.ndarray
        Approximate solution with the same shape as b.

    Raises ValueError
        If the dimensions of A, b, or x0 are incompatible or if alg is invalid.
    RuntimeWarning
        If the solution does not converge within the maximum iterations.
    """

    # Convert inputs to numpy arrays
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
 
    if A.ndim != 2: #check dimensions
        raise ValueError("Matrix A must be 2D.")
    m, n = A.shape # assign m to rows and n to columns
    if m != n: #checks if square
        raise ValueError("Matrix A must be square.")

    if b.ndim == 1: # if 1d treat as vector instead of matrix
        b = b.reshape(-1, 1)  # treat as column vector

    if b.shape[0] != n: # Check if the rows match the number of rows/columns in A
        raise ValueError("The number of rows in A must match the number of rows in b.")

    # --- Initialize x0 ---
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x0 = np.array(x0, dtype=float)
        if x0.ndim == 1: #checks dimensions
            x0 = x0.reshape(-1, 1) #Converts to column (transposees it)
        if x0.shape[0] != n: 
            raise ValueError("Initial guess x0 has incompatible dimensions.")
        if x0.shape[1] == 1 and b.shape[1] > 1:
            x = np.tile(x0, (1, b.shape[1]))
        elif x0.shape == b.shape:
            x = x0.copy()
        else:
            raise ValueError("Initial guess x0 has incompatible shape with b.")
    
    # --- Algorithm selection ---
    alg = alg.strip().lower()
    if alg not in ['seidel', 'jacobi']:
        raise ValueError("alg must be either 'seidel' or 'jacobi'.")

    # --- Iterative Solver ---
    max_iter = 10000
    D = np.diag(A)
    if np.any(D == 0):
        raise ValueError("Matrix A has zero(s) on the diagonal.")

    L = np.tril(A, -1)
    U = np.triu(A, 1)
       
    for k in range(max_iter):
        x_old = x.copy()

        if alg == 'jacobi':
            x = (b - ((L + U) @ x_old)) / D[:, None]
        elif alg == 'seidel':
            for i in range(n):
                x[i, :] = (b[i, :] - L[i, :] @ x - U[i, :] @ x_old) / D[i]

        # --- Convergence Check ---
        rel_error = np.linalg.norm(x - x_old) / (np.linalg.norm(x) + 1e-15)
        if rel_error < tol:
            return np.squeeze(x)

    # If we reach here, we didn't converge
    warnings.warn("Solution did not converge within max iterations.", RuntimeWarning)
    return np.squeeze(x)

def spline_function(xd, yd, order=3):
    """
    function that generates a spline function given two vectors x and y of data.
    xd: array_like of float data increasing in value
    yd: array_like of float data with the same shape as xd
    ord: order is an (optional) integer with possible values 1, 2, or 3 (default=3)
    returns: function that takes one parameter (a float or array_like of float) and returns the interpolated y value(s)
    """
    xd = np.asarray(xd, dtype=float).flatten() #convert into array and use flatten to ensure of a 1d vector. 
    yd = np.asarray(yd, dtype=float).flatten()

    if xd.shape[0] != yd.shape[0]: #Confirm they are the same dimensions
        raise ValueError("xd and yd must have the same length.")

    if len(np.unique(xd)) != xd.shape[0]: #Combines non-unique value into one space since it must be increasing, then calls error
        raise ValueError("xd contains repeated values.")

    if not np.allclose(np.sort(xd), xd):
        raise ValueError("xd must be strictly increasing.") #checks if the sorted version of the array is equal to the non-sorted (must be increasing)
   
    if order not in (1, 2, 3):
        raise ValueError("order must be 1, 2, or 3.")
    xmin = xd[0] #sets to first int
    xmax = xd[-1] #Sets to furthermost right int
    n = len(xd) #Amount of ints in xd
    h = np.diff(xd) #h is an array with the differences between the points of xd
    
    if order == 1: #Linear spline
        slopes = np.diff(yd) / h #Rise over Run, results in an array
        def f(x):
            x = np.asarray(xd, dtype=float) #Converts x into array
            if np.any((x < xmin) | (x > xmax)): #Prevents extrapolation outside the data range.
                raise ValueError(f"Input out of range: xmin={xmin}, xmax={xmax}")
            y = np.empty_like(x, dtype=float) #Create output array
            for i in range(n -1):
                mask = (x >= xd[i]) & (x <= xd[i + 1])
                y[mask] = yd[i] + slopes[i] * (x[mask] - xd[i])
            return y
        return f
    elif order == 2: #Quadratic spline 
        def f(x):     
            x = np.asarray(x)
            if np.any(x < xd[0]) or np.any(x > xd[-1]): #Prevents extrapolation outside the data range.
                raise ValueError(f"x is out of bounds: [{xd[0]}, {xd[-1]}]")
            y = np.empty_like(x, dtype=float) #Create output array
            for i in range(n-2):
                xi = xd[i:i+3]
                yi = yd[i:i+3]
                A = np.vstack([xi**2, xi, np.ones_like(xi)]).T #Build vandermond matrix and solve for a,b,c
                coeff = solve(A, yi)
                mask = (x >= xi[0]) & (x < xi[2]) if i < n - 3 else (x >= xi[0]) & (x <= xi[2])
                y[mask] = coeff[0]*x[mask]**2 + coeff[1]*x[mask] + coeff[2] #Evaluate polynomial
            return y
        return f
    elif order == 3:
        def f(x):
            x = np.asarray(x, dtype=float)
            if np.any(x < xd[0]) or np.any(x > xd[-1]): #Prevents extrapolation outside the data range.
                raise ValueError(f"x is out of bounds: [{xd[0]}, {xd[-1]}]")
            y = np.empty_like(x, dtype=float) #Create output array
            for i in range(n - 3): # Loop through segments Loop through all segments Each segment uses 4 points to fit a cubic polynomial.
                xi = xd[i:i+4]   # 4 points for cubic
                yi = yd[i:i+4]
                A = np.vstack([xi**3, xi**2, xi, np.ones_like(xi)]).T # Build Vandermonde matrix to solve for a,b,c,d
                coeff = solve(A, yi)  # coeff = [a, b, c, d]
                if i < n - 4: # Mask: x values in this segment
                    mask = (x >= xi[0]) & (x < xi[3])
                else:
                    mask = (x >= xi[0]) & (x <= xi[3])
                y[mask] = coeff[0]*x[mask]**3 + coeff[1]*x[mask]**2 + coeff[2]*x[mask] + coeff[3]   # Evaluate cubic polynomial on masked x values
            return y
        return f

    
