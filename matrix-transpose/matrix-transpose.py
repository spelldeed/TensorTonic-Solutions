import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    # return np.transpose(A)
    r = len(A[0])
    c = len(A)

    output_arr = np.zeros((r,c))
    for i in range(0,r):
        for j in range(0,c):
           output_arr[i][j] = A[j][i]
    
    return output_arr
