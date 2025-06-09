from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from numba import njit
from typeguard import typechecked


@njit
def logsumexp1dNumba(arr):
    max_val = -np.inf
    n = arr.shape[0]
    for i in range(n):
        if arr[i] > max_val:
            max_val = arr[i]
    s = 0.0
    for i in range(n):
        s += np.exp(arr[i] - max_val)
    return max_val + np.log(s)


@njit
def logsumexp2dNumba(arr, axis):
    nRows, nCols = arr.shape

    # Sum over rows (axis 0)
    if axis == 0:
        result = np.empty(nCols)
        for j in range(nCols):
            max_val = -np.inf
            for i in range(nRows):
                if arr[i, j] > max_val:
                    max_val = arr[i, j]
            s = 0.0
            for i in range(nRows):
                s += np.exp(arr[i, j] - max_val)
            result[j] = max_val + np.log(s)
    # Sum over columns (axis 1)
    elif axis == 1:
        result = np.empty(nRows)
        for i in range(nRows):
            max_val = -np.inf
            for j in range(nCols):
                if arr[i, j] > max_val:
                    max_val = arr[i, j]
            s = 0.0
            for j in range(nCols):
                s += np.exp(arr[i, j] - max_val)
            result[i] = max_val + np.log(s)

    else:
        raise ValueError("Invalid axis. Must be 0 or 1.")

    return result


@njit
def laplaceSmoothing(logMatrix, logAlpha):
    nRows, nCols = logMatrix.shape
    # adding alpha via logsumexp per entry
    u = np.empty_like(logMatrix)
    for i in range(nRows):
        for j in range(nCols):
            b = logMatrix[i, j]
            a = logAlpha
            if b > a:
                m = b
                u[i, j] = m + np.log(1 + np.exp(a - m))
            else:
                m = a
                u[i, j] = m + np.log(1 + np.exp(b - m))

    # computing row-wise normalizer Z[i] = logsumexp(u[i, :])
    Z = np.empty(nRows)
    for i in range(nRows):
        Z[i] = logsumexp1dNumba(u[i, :])

    # Step 3: subtract normalizer from each row
    smoothed = np.empty_like(u)
    for i in range(nRows):
        for j in range(nCols):
            smoothed[i, j] = u[i, j] - Z[i]

    return smoothed


@typechecked
def smoothing(P: Optional[npt.NDArray[np.float64]] = None, A: Optional[npt.NDArray[np.float64]] = None,
              B: Optional[npt.NDArray[np.float64]] = None) -> Tuple[
        Optional[npt.NDArray[np.float64]], Optional[npt.NDArray[np.float64]], Optional[npt.NDArray[np.float64]]
    ]:
    sP: Optional[npt.NDArray[np.float64]] = None
    sA: Optional[npt.NDArray[np.float64]] = None
    sB: Optional[npt.NDArray[np.float64]] = None
    logPseudo = np.log(1e-30)

    if P is not None:
        sP = laplaceSmoothing(np.array([P]), logPseudo)[0]
    if A is not None:
        sA = laplaceSmoothing(A, logPseudo)
    if B is not None:
        sB = laplaceSmoothing(B, logPseudo)

    return sP, sA, sB
