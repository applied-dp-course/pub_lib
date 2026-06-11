import numpy as np
from scipy.optimize import linprog


def int_prog_reconstruction(subsets: np.ndarray, responses: np.ndarray) -> np.ndarray:
    """
    Solves the binary vector reconstruction problem by minimizing the L-infinity error.

    Args:
        subsets: A m x n binary matrix, where each row represents a subset of the data
        responses: A m-dimensional vector of real numbers

    Returns:
        A n-dimensional integer vector (0s and 1s) that minimizes the L-infinity reconstruction error

    The function works by:
    1. Enumerating all possible n-bit binary vectors
    2. Computing the L-infinity error between subset sums and given responses
    3. Selecting the binary vector that minimizes this error
    """
    m, n = subsets.shape

    # Generate powers of 2 for converting integers to n-bit binary vectors
    powers_of_two = 1 << np.arange(n - 1, -1, -1)

    # For each possible n-bit number (0 to 2^n - 1):
    # `(i & powers_of_two) > 0` is a vectorized bitwise AND operation which translates i to a binary vector
    # `subsets.dot(vec) - responses`` is the diff vector between the correct responses if vec is the true data and the produced responses
    # `np.linalg.norm(diff, ord=np.inf)` is the l infinity norm of the diff vector
    error_vec = np.array(
        [
            np.linalg.norm(subsets.dot((i & powers_of_two) > 0) - responses, ord=np.inf)
            for i in range(2**n)
        ]
    )

    # Find the number that gave minimum error (if there are multiple, return the first); this is effectively argmin
    min_error_index = np.argwhere(error_vec == np.min(error_vec))[0][0]

    # Convert that number back to binary vector and cast to an integer type
    return ((min_error_index & powers_of_two) > 0).astype(int)


def lin_reg_reconstruction(subsets: np.ndarray, responses: np.ndarray) -> np.ndarray:
    """
    Approximates the binary vector reconstruction problem using linear regression
    followed by rounding/clipping to minimize the L2 error.

    Args:
        subsets: A m x n binary matrix, where each row represents a subset of the data
        responses: A m-dimensional vector of real numbers

    Returns:
        A n-dimensional binary vector (as integers) that approximates the minimum L2 error solution
    """
    m, n = subsets.shape

    # Handle empty arrays
    if m == 0 or n == 0:
        return np.array([], dtype=int)

    # Get a continuous solution of the linear regression
    continuous_solution = np.linalg.lstsq(subsets, responses, rcond=None)[0]

    # Clip values to [0,1] range and then round to the nearest binary value
    # Due to the binary nature of the problem, this is equivalent to the truth value of the > 0.5 condition
    return (continuous_solution > 0.5).astype(int)


def lin_prog_reconstruction(subsets: np.ndarray, responses: np.ndarray) -> np.ndarray:
    """
    Reconstructs a binary vector using linear programming by minimizing the L1 norm
    of the reconstruction error.

    This function formulates the binary reconstruction problem as:
    minimize    sum(slack variables)
    subject to  subsets * x - slack <= responses
                -subsets * x - slack <= -responses
                0 <= x <= 1
                0 <= slack <= n

    Args:
        subsets: A m x n binary matrix, where each row represents a subset of the data
        responses: A m-dimensional vector of real numbers

    Returns:
        A n-dimensional binary vector that approximates the solution
    """

    # Convert subsets to integer type
    subsets = subsets.astype(int)
    # Get dimensions of the problem
    n = subsets.shape[1]  # number of variables
    m = subsets.shape[0]  # number of constraints/measurements

    # Construct objective function coefficients
    # [0, ..., 0, 1, ..., 1] where we have n zeros (for x) and m ones (for slack variables)
    c = np.concatenate((np.zeros(n), np.ones(m)))

    # Construct inequality constraints matrix
    # For constraints: |Ax - b| ≤ slack
    # Which is equivalent to:
    # Ax - slack ≤ b
    # -Ax - slack ≤ -b
    A_1 = np.hstack((subsets, -np.eye(m)))  # For upper bound: Ax - slack ≤ b
    A_2 = np.hstack((-subsets, -np.eye(m)))  # For lower bound: -Ax - slack ≤ -b
    A_ub = np.vstack((A_1, A_2))  # Combined constraints matrix

    # Construct inequality constraints bounds
    b_ub = np.concatenate((responses, -responses))

    # Construct variable bounds
    # x variables are bounded by [0,1]
    # slack variables are bounded by [0,n]
    lhs_bounds = np.zeros((n + m, 1))  # Lower bounds
    rhs_bounds = np.vstack(
        (  # Upper bounds
            np.ones((n, 1)),  # For x variables: ≤ 1
            n * np.ones((m, 1)),  # For slack variables: ≤ n
        )
    )
    bounds = np.hstack([lhs_bounds, rhs_bounds])

    # Solve linear program
    res = linprog(
        c=c,  # Objective function coefficients
        A_ub=A_ub,  # Inequality constraints matrix
        b_ub=b_ub,  # Inequality constraints bounds
        bounds=bounds.tolist(),  # Variable bounds
        method='highs',  # Using the more modern HiGHS solver
    )

    if not res.success:
        raise RuntimeError(f"Linear programming optimization failed: {res.message}")

    # Extract solution and round to binary
    return np.rint(res.x[:n]).astype(int)
