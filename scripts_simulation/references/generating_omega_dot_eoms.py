# %% 
# Symbolic Operations with SymPy
# 
# This notebook will use SymPy to perform symbolic operations 
# to solve for the Euler angle rates and angular velocity

# %% 
# Imports
import sympy as sp
from IPython.display import display, Math

# %% 
# Define the symbolic variables
phi, theta, psi = sp.symbols('phi theta psi')
phi_dot, theta_dot, psi_dot = sp.symbols('phi_dot theta_dot psi_dot')
omega_x, omega_y, omega_z = sp.symbols('omega_x omega_y omega_z')

# %% 
# Define the rotation matrices
R_z_0_to_1 = sp.Matrix([
    [sp.cos(psi), sp.sin(psi), 0],
    [-sp.sin(psi), sp.cos(psi), 0],
    [0,            0,           1]
])
R_z_1_to_0 = R_z_0_to_1.inv()

R_y_1_to_2 = sp.Matrix([
    [sp.cos(theta), 0, -sp.sin(theta)],
    [0,              1, 0],
    [sp.sin(theta), 0, sp.cos(theta)]
])
R_y_2_to_1 = R_y_1_to_2.inv()

R_x_2_to_3 = sp.Matrix([
    [1, 0,            0           ],
    [0, sp.cos(phi), sp.sin(phi)],
    [0, -sp.sin(phi), sp.cos(phi)]
])
R_x_3_to_2 = R_x_2_to_3.inv()

# %% 
# Define phi, theta, psi in all frames
phi_in_2 = sp.Matrix([phi_dot, 0, 0])
theta_in_1 = sp.Matrix([0, theta_dot, 0])
psi_in_0 = sp.Matrix([0, 0, psi_dot])

phi_in_3 = R_x_2_to_3 * phi_in_2
phi_in_1 = R_y_2_to_1 * phi_in_2
phi_in_0 = R_z_1_to_0 * phi_in_1

theta_in_0 = R_z_1_to_0 * theta_in_1
theta_in_2 = R_y_1_to_2 * theta_in_1
theta_in_3 = R_x_2_to_3 * theta_in_2

psi_in_1 = R_z_0_to_1 * psi_in_0
psi_in_2 = R_y_1_to_2 * psi_in_1
psi_in_3 = R_x_2_to_3 * psi_in_2

# %% 
# Define omega in inertial and body frames
omega_in_0 = phi_in_0 + theta_in_0 + psi_in_0
omega_in_3 = phi_in_3 + theta_in_3 + psi_in_3

# %% 
# Find the transformation matrices omega_in_0 and omega_in_3
angular_rates = sp.Matrix([phi_dot, theta_dot, psi_dot])

omega_in_0_matrix = omega_in_0.jacobian(angular_rates)
omega_in_0_matrix_simplified = sp.simplify(omega_in_0_matrix)

omega_in_3_matrix = omega_in_3.jacobian(angular_rates)
omega_in_3_matrix_simplified = sp.simplify(omega_in_3_matrix)

# %% 
# Display the transformations
display(Math(r"\begin{bmatrix} \omega_x \\ \omega_y \\ \omega_z \end{bmatrix}_{0} = " + sp.latex(omega_in_0_matrix_simplified * angular_rates)))
display(Math(r"= " + sp.latex(omega_in_0_matrix_simplified) + r"\begin{bmatrix} \dot{\phi} \\ \dot{\theta} \\ \dot{\psi} \end{bmatrix}"))

# display(Math(r"\begin{bmatrix} \omega_x \\ \omega_y \\ \omega_z \end{bmatrix}_{3} = " + sp.latex(omega_in_3_matrix_simplified * angular_rates)))
# display(Math(r"= " + sp.latex(omega_in_3_matrix_simplified) + r"\begin{bmatrix} \dot{\phi} \\ \dot{\theta} \\ \dot{\psi} \end{bmatrix}"))

# %% 
# Solve for angular rates
omega = sp.Matrix([omega_x, omega_y, omega_z])

omega_in_0_matrix_inv = omega_in_0_matrix_simplified.inv()
omega_in_0_matrix_inv_simplified = sp.trigsimp(omega_in_0_matrix_inv)
angular_rates_0 = omega_in_0_matrix_inv_simplified * omega
angular_rates_0_simplified = sp.trigsimp(angular_rates_0)

omega_in_3_matrix_inv = omega_in_3_matrix_simplified.inv()
omega_in_3_matrix_inv_simplified = sp.trigsimp(omega_in_3_matrix_inv)
angular_rates_3 = omega_in_3_matrix_inv_simplified * omega
angular_rates_3_simplified = sp.trigsimp(angular_rates_3)

# %% 
# Display the angular rates
display(Math(r"\begin{bmatrix} \dot{\phi} \\ \dot{\theta} \\ \dot{\psi} \end{bmatrix} = " + sp.latex(angular_rates_0_simplified)))
display(Math(r"= " + sp.latex(omega_in_0_matrix_inv_simplified) + r"\begin{bmatrix} \omega_x \\ \omega_y \\ \omega_z \end{bmatrix}_{0}"))

# display(Math(r"\begin{bmatrix} \dot{\phi} \\ \dot{\theta} \\ \dot{\psi} \end{bmatrix} = " + sp.latex(angular_rates_3_simplified)))
# display(Math(r"= " + sp.latex(omega_in_3_matrix_inv_simplified) + r"\begin{bmatrix} \omega_x \\ \omega_y \\ \omega_z \end{bmatrix}_{3}"))

# %%