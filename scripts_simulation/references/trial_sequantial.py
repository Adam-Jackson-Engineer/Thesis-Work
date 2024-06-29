import numpy as np

x = np.array([[1], [0], [0], [0], [1], [0]])
y = np.zeros_like(x)
z = x!=y

num_diff = np.count_nonzero(z)

for i in range(num_diff):
    pass

m = np.where(z)
print(m)
print(m[0][1])
