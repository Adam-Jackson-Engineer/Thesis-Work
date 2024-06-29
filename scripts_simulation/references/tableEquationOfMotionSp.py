import sympy as sp
from sympy.interactive import printing
printing.init_printing(use_latex=True)

def NumDiff(r,t):
    rdot = sp.diff(r, t)
    return rdot

def rot_x(phi):
    R_x = sp.Matrix([
        [1, 0, 0],
        [0, sp.cos(phi), sp.sin(phi)],
        [0, -sp.sin(phi), sp.cos(phi)]])
    return R_x

def rot_y(tht):    
    R_y = sp.Matrix([
        [sp.cos(tht), 0, -sp.sin(tht)],
        [0, 1, 0],
        [sp.sin(tht), 0, sp.cos(tht)]])
    return R_y

def rot_z(psi):
    R_z = sp.Matrix([
        [sp.cos(psi), sp.sin(psi), 0],
        [-sp.sin(psi), sp.cos(psi), 0],
        [0, 0, 1]])
    return R_z

def bodyVecDeriv(r, w, t):
    rdotrel = sp.diff(r, t)
    rdot = rdotrel + w.cross(r)
    return rdot

# Generate values using list comprehensions and convert directly to numpy arrays
t = sp.Symbol('t')
xI = sp.Function('xI')(t)
yI = sp.Function('yI')(t)
zI = sp.Function('zI')(t)
xB = sp.Function('xB')(t)
yB = sp.Function('yB')(t)
zB = sp.Function('zB')(t)
phiI = sp.Function('phiI')(t)
thtI = sp.Function('thtI')(t)
psiI = sp.Function('psiI')(t)

posI = sp.Matrix([[xI],[yI],[zI]])
posB = sp.Matrix([[xB],[yB],[zB]])
angI = sp.Matrix([[phiI],[thtI],[psiI]])

# %%
# First derivatives
posIdot = NumDiff(posI,t)
angIdotrel = NumDiff(angI,t)

# %%
# Second derivatives
posIdotdot = NumDiff(posIdot,t)

R_x = rot_x(angI[0])
R_y = rot_y(angI[1])
R_z = rot_z(angI[2])

# Combined rotation matrix
Rz = R_z
Ryz = R_y @ R_z
Rxyz = R_x @ R_y @ R_z

# %%
# Position in body frame
# posB = Rxyz @ posI

# %%
# Angular velocity in inertial frame
angIdot = sp.Matrix([[0.0],[0.0],[angIdotrel[2]]]) + \
            Rz.T @ sp.Matrix([[0.0],[angIdotrel[1]],[0.0]]) + \
            Ryz.T @ sp.Matrix([[angIdotrel[0]],[0.0],[0.0]])
angIdot = sp.simplify(angIdot)

# %%
# Angular velocity in body frame
angBdot = Rxyz @ angIdot
angBdot = sp.simplify(angBdot)

# %%
# Velocity and Acceleration in Body frame
posBdot = bodyVecDeriv(posB,angBdot,t)
posBdot = sp.simplify(posBdot)
posBdotdot = bodyVecDeriv(posBdot,angBdot,t)
posBdotdot = sp.simplify(posBdotdot)

# %%
# Display the equations
# posBdotdot_simp = sp.simplify(posBdotdot)

# %%
# Write the LaTeX code to a text file
latex_code = sp.latex(posBdot)
with open('posBdot.tex', 'w') as file:
    file.write(latex_code)
print("DONE")
# %%
# Write the LaTeX code to a text file
posBdotdot = sp.simplify(posBdotdot)
latex_code = sp.latex(posBdotdot)
with open('posBdotdot.tex', 'w') as file:
    file.write(latex_code)
print("DONE")

# %%