import numpy as np
import scripts_simulation.table_parameters as P
import matplotlib.pyplot as plt
from scripts_simulation.signal_generator import signalGenerator

def genSignal(sig,time):
    r = np.array([sig.sin(t) for t in time])
    return r

def NumDiffSameSize(r,dt):
    rdot = np.diff(r, axis = 0) / dt
    rdot = np.vstack((rdot, rdot[-1]))
    return rdot

def bodyVecDeriv(r, w, Ts):
    rdotrel = np.diff(r, axis=0)/Ts
    rdotrel = np.vstack((rdotrel, rdotrel[-1]))
    
    rdot = np.zeros(np.shape(rdotrel))
    for i in range(len(rdot)):
        rdot[i] = rdotrel[i] + np.cross(w[i],r[i])
    return rdot

def rot_x(phi):
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(phi), np.sin(phi)],
        [0, -np.sin(phi), np.cos(phi)]])
    return R_x

def rot_y(tht):    
    R_y = np.array([
        [np.cos(tht), 0, -np.sin(tht)],
        [0, 1, 0],
        [np.sin(tht), 0, np.cos(tht)]])
    return R_y

def rot_z(psi):
    R_z = np.array([
        [np.cos(psi), np.sin(psi), 0],
        [-np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]])
    return R_z

# Time vector initialization
t = P.t_start
time_vector = np.arange(P.t_start, P.t_end, P.Ts)  # Assuming P.ts is the sampling time

# Generate values using list comprehensions and convert directly to numpy arrays
xI = genSignal(signalGenerator(amplitude=0.4, frequency=0.9), time_vector)
yI = genSignal(signalGenerator(amplitude=0.5, frequency=0.8), time_vector)
zI = genSignal(signalGenerator(amplitude=0.6, frequency=0.7), time_vector)
phiI = genSignal(signalGenerator(amplitude=0.7, frequency=0.6), time_vector)
thtI = genSignal(signalGenerator(amplitude=0.8, frequency=0.5), time_vector)
psiI = genSignal(signalGenerator(amplitude=0.9, frequency=0.4), time_vector)

posI = np.concatenate((xI[:, np.newaxis], yI[:, np.newaxis], zI[:, np.newaxis]), axis=1)
angI = np.concatenate((phiI[:, np.newaxis], thtI[:, np.newaxis], psiI[:, np.newaxis]), axis=1)

# First derivatives
posIdot = NumDiffSameSize(posI,P.Ts)
angIdotrel = NumDiffSameSize(angI,P.Ts)

# Second derivatives
posIdotdot = NumDiffSameSize(posIdot,P.Ts)

posB = np.zeros(np.shape(posI))
angIdot = np.zeros(np.shape(angIdotrel))
angBdot = np.zeros(np.shape(angIdot))

for i in range(len(xI)):
    R_x = rot_x(angI[i][0])
    R_y = rot_y(angI[i][1])
    R_z = rot_z(angI[i][2])

    # Combined rotation matrix
    Rz = R_z
    Ryz = R_y @ R_z
    Rxyz = R_x @ R_y @ R_z
    
    posB[i] = Rxyz @ posI[i].T
    angIdot[i] = np.squeeze(np.array([[0.0],[0.0],[angIdotrel[i][2]]]) + \
                Rz.T @ np.array([[0.0],[angIdotrel[i][1]],[0.0]]) + \
                Ryz.T @ np.array([[angIdotrel[i][0]],[0.0],[0.0]]))
                # psidot is the in origional frame
                # thtdot is the the z rotated frame
                # phi dot is in the z rot y rot frame
    angBdot[i] = Rxyz @ angIdot[i]

posBdot = bodyVecDeriv(posB,angBdot,P.Ts)
posBdotdot = bodyVecDeriv(posBdot,angBdot,P.Ts)

# Calculate the magnitude of each position vector in posBdot
posIdotMag = np.linalg.norm(posIdotdot, axis=1)
posBdotMag = np.linalg.norm(posBdotdot, axis=1)

# Plot the magnitudes over time in the first subplot
plt.figure(figsize=(8, 6))  # Adjusts the figure size to accommodate two subplots
plt.subplot(2, 1, 1)  # This creates the first subplot (2 rows, 1 column, first plot)
plt.plot(time_vector, posIdotMag, label='Magnitude of posIdot', linestyle='-')  # Dashed line for posIdot
plt.plot(time_vector, posBdotMag, label='Magnitude of posBdot', linestyle='--')  # Solid line for posBdot
plt.xlabel('Time (s)')
plt.ylabel('Magnitude')
plt.title('Magnitude of Position Derivatives over Time')
plt.xlim([0, 5])  # Set the x-axis limit
plt.legend()
plt.grid(True)

# Add a subplot for phiI, thtI, and psiI
plt.subplot(2, 1, 2)  # This creates the second subplot (2 rows, 1 column, second plot)
plt.plot(time_vector, phiI, label='phiI (Roll)', linestyle='-.')  # Dot-dash line for phiI
plt.plot(time_vector, thtI, label='thtI (Pitch)', linestyle=':')  # Dotted line for thtI
plt.plot(time_vector, psiI, label='psiI (Yaw)', linestyle='-')  # Solid line for psiI
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.title('Roll, Pitch, and Yaw Angles over Time')
plt.xlim([0, 5])  # Set the x-axis limit
plt.legend()
plt.grid(True)

plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area.
plt.show()