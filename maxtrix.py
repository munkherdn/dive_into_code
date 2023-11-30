#problem 1 calculate matrix by hand
from IPython.display import display, Image

image_path = 'Screenshot 2023-11-30 at 19.40.34.png'
display(Image(filename=image_path))

#problem 2
import numpy as np

a_ndarray = np.array([[-1, 2, 3], [4, -5, 6], [7, 8, -9]])
b_ndarray = np.array([[0, 2, 1], [0, 2, -8], [2, 9, -1]])

result = np.matmul(a_ndarray, b_ndarray)
print(result)

#problem #3
def calculate_element(a, b, i, j):
    element = 0
    for k in range(len(a[0])):
        element += a[i, k] * b[k, j]
    return element

element_00 = calculate_element(a_ndarray, b_ndarray, 0, 0)
print(element_00)

#problem 4
def matrix_multiply(a, b):
    c = np.zeros((len(a), len(b[0])))
    for i in range(len(a)):
        for j in range(len(b[0])):
            c[i, j] = calculate_element(a, b, i, j)
    return c

result_scratch = matrix_multiply(a_ndarray, b_ndarray)
print(result_scratch)

#problem 5
def matrix_multiply_safe(a, b):
    if a.shape[1] != b.shape[0]:
        print("Error: Matrix multiplication is not defined for the given matrices.")
        return None
    else:
        return matrix_multiply(a, b)

# Example usage:
result_safe = matrix_multiply_safe(d_ndarray, e_ndarray)
if result_safe is not None:
    print(result_safe)

#problem 6
# Transpose matrix D
d_transposed = np.transpose(d_ndarray)

# Check if the number of columns of transposed D is equal to the number of rows of E
if d_transposed.shape[1] != e_ndarray.shape[0]:
    print("Error: Matrix multiplication is not defined for the transposed matrices.")
else:
    # Calculate the matrix product
    result_transposed = np.matmul(d_transposed, e_ndarray)

    # Alternatively, you can use d_transposed @ e_ndarray
    # result_transposed = d_transposed @ e_ndarray

    print(result_transposed)
