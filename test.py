import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([1, 2, 3, 4, 5])

c = a / b + 3

matrix = np.zeros((5, 5), dtype=int)

# 填充上对角线 (即主对角线的上方)
np.fill_diagonal(matrix[:, 1:], [1, 2, 3, 4])

b = b.reshape(-1, 1)
print(a + b)

d = np.arange(0, 10, 1)

e = a * np.sqrt(a * c - (np.sin(d.reshape(-1, 1)) ** 2))
print(e.shape[0])

a0 = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
print(a.shape)
print(a0.shape)
print(b[-3])
