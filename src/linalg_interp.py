# File: src/lab_02/linalg_interp.py

import numpy as np
import warnings

def gauss_iter_solve(A, b, x0, tol, alg):
    """
    Solve the linear system A x = b using the Gauss-Seidel or Jacobi iterative method.

    Parameters
    ----------
    A : array_like
        Coefficient matrix (must be square)
    b : array_like
        Right-hand-side vector or matrix
    x0 : array_like, optional
        Initial guess for the solution. Default is None (uses zeros)
    tol : float, optional
        Relative error tolerance for convergence. Default is 1e-8
    alg : str, optional
        Iteration algorithm: 'seidel' (default) or 'jacobi'. Case-insensitive.

    Returns
    -------
    x : numpy.ndarray
        Approximate solution with the same shape as b.

    Raises
    ------
    ValueError
        If the dimensions of A, b, or x0 are incompatible or if alg is invalid.
    RuntimeWarning
        If the solution does not converge within the maximum iterations.
    """

    # Convert inputs to numpy arrays
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    # --- Validate dimensions ---
    if A.ndim != 2:
        raise ValueError("Matrix A must be 2D.")
    m, n = A.shape
    if m != n: #checks if square
        raise ValueError("Matrix A must be square.")

    if b.ndim == 1:
        b = b.reshape(-1, 1)  # treat as column vector

    if b.shape[0] != n:
        raise ValueError("The number of rows in A must match the number of rows in b.")

    # --- Initialize x0 ---
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x0 = np.array(x0, dtype=float)
        if x0.ndim == 1: #checks dimensions
            x0 = x0.reshape(-1, 1)
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




