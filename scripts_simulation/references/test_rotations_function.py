from scipy.spatial.transform import Rotation
import numpy as np

print()
quat = [-0.00257268913111,	0.001759615195328,	-0.045248331300454, 0.998970478941863	
]

# Convert quaternion to rotation matrix
rotation = Rotation.from_quat(quat)
print("Rotation matrix:")
print(rotation.as_matrix())
print()

# Convert to Euler angles
euler_angles = rotation.as_euler('xyz', degrees=True)
print("Euler angles (xyz):")
print(euler_angles)
print()

# Convert to Euler angles
euler_angles = rotation.as_euler('zyx', degrees=True)
print("Euler angles (zyx):")
print(euler_angles)
print()
