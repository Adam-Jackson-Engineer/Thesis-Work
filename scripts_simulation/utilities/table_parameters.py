"""
table_parameters.py

Contains parameters for the table simulation.
"""

import os
import numpy as np
from scipy.spatial.transform import Rotation

from utilities.signal_generator import SignalGenerator

# Physical dimensions of the table
TABLE_MASS = 24.551      # kg (Mass)
TABLE_LENGTH = 1.2192    # m  (Length) (pos X forward)
TABLE_WIDTH = 0.6064     # m  (Width) (pos Y Left)
TABLE_HEIGHT = 0.1921    # m  (Height) (pos Z up)
GRAVITY = 9.81           # m/s^2

# Dimensions of the workspace
ROOM_LENGTH = 6          # m (Length of Room) (X)
ROOM_WIDTH = 5           # m (Width of Room) (Y)

# Applied force distance (handle location)
OFFSET_WIDTH = 0.254     # m (Offset from width) (Offset in Y direction)
OFFSET_WIDTH_CENTER = 0    # m (Center of handles is in line with center of mass)
OFFSET_HEIGHT = 0        # m (Assumes forces in X are applied in line with CoM)
OFFSET_LENGTH = TABLE_LENGTH / 2  # m (Offset from length) (Assumes force applied at half table length away)

LEADER_FORCE_LOCATION = np.array([[-OFFSET_LENGTH], [OFFSET_WIDTH_CENTER], [OFFSET_HEIGHT]])
FOLLOWER_FORCE_LOCATION = np.array([[OFFSET_LENGTH], [OFFSET_WIDTH_CENTER], [OFFSET_HEIGHT]])

# Calculated moments of inertia (assumes uniformly distributed rectangular prism)
INERTIA_XX = (1 / 12) * TABLE_MASS * (TABLE_WIDTH ** 2 + TABLE_HEIGHT ** 2)
INERTIA_YY = (1 / 12) * TABLE_MASS * (TABLE_LENGTH ** 2 + TABLE_HEIGHT ** 2)
INERTIA_ZZ = (1 / 12) * TABLE_MASS * (TABLE_LENGTH ** 2 + TABLE_WIDTH ** 2)

# Inertia matrix of table
INERTIA_MATRIX = np.array([
    [INERTIA_XX, 0, 0], 
    [0, INERTIA_YY, 0], 
    [0, 0, INERTIA_ZZ]
])

# Initial conditions (expressed in body frame, relative to inertial frame)
# The inertial frame is located on the floor at the starting location
X_0 = 0.0
Y_0 = 0.0
Z_0 = 0.5        # m (Assumes table starts in center of room 0.5 m off ground)
PHI_0 = 0.0
THETA_0 = 0.0
PSI_0 = 0.0       # The table frame is oriented as a 180 deg rotation from the inertial frame
X_DOT_0 = 0.0
Y_DOT_0 = 0.0
Z_DOT_0 = 0.0
OMEGA_X_0 = 0.0
OMEGA_Y_0 = 0.0
OMEGA_Z_0 = 0.0

INITIAL_STATES = np.array([[X_0], [Y_0], [Z_0], [PHI_0], [THETA_0], [PSI_0], [X_DOT_0], [Y_DOT_0], [Z_DOT_0], [OMEGA_X_0], [OMEGA_Y_0], [OMEGA_Z_0]])

# Force max values (95th percentile forces from the dyads, body frame)
FORCE_X_MAX_LEADER = 46.0    # N
FORCE_Y_MAX_LEADER = 34.5    # N
FORCE_Z_MAX_LEADER = 152.0   # N 
TORQUE_PHI_MAX_LEADER = 9.9     # Nm
TORQUE_THETA_MAX_LEADER = 4.4     # Nm
TORQUE_PSI_MAX_LEADER = 24.5    # Nm

FORCE_X_MAX_FOLLOWER = 38.0  # N
FORCE_Y_MAX_FOLLOWER = 27.0  # N
FORCE_Z_MAX_FOLLOWER = 152.0 # N 
TORQUE_PHI_MAX_FOLLOWER = 9.9   # Nm
TORQUE_THETA_MAX_FOLLOWER = 4.4   # Nm
TORQUE_PSI_MAX_FOLLOWER = 7.5   # Nm

# Simulation parameters
T_START = 0.0           # Start time of simulation
T_END = 15.0            # End time of simulation
T_STEP = 0.005           # Sample time for simulation, based on real data time
T_PLOT = 0.05           # The plotting and animation is updated at this rate

X_STEP = 1.0
Y_STEP = 1.0
Z_STEP = 0.5
PHI_STEP = np.pi / 2
THETA_STEP = np.pi / 4
PSI_STEP = np.pi / 2

# Reference signals
signal_force_x = SignalGenerator(amplitude=200, frequency=1.0, y_offset=0.5)
signal_force_y = SignalGenerator(amplitude=20, frequency=0.1, y_offset=0.5)
signal_force_z = SignalGenerator(amplitude=20, frequency=0.1, y_offset=0.5)
signal_torque_phi = SignalGenerator(amplitude=5, frequency=1.0)
signal_torque_theta = SignalGenerator(amplitude=5, frequency=1.0)
signal_torque_psi = SignalGenerator(amplitude=5, frequency=1.0)

signal_reference_x = SignalGenerator(amplitude=1.0, frequency=0.1)
signal_reference_y = SignalGenerator(amplitude=1.0, frequency=0.1)
signal_reference_z = SignalGenerator(amplitude=0.25, frequency=0.1)
signal_reference_phi = SignalGenerator(amplitude=np.pi/2, frequency=0.1)
signal_reference_theta = SignalGenerator(amplitude=np.pi/4, frequency=0.1)
signal_reference_psi = SignalGenerator(amplitude=np.pi/2, frequency=0.1)    

# Reference positions
REFERENCE_REST =        np.array([[X_0],            [Y_0],              [Z_0],          [PHI_0],                [THETA_0],                  [PSI_0]])
REFERENCE_TX =          np.array([[X_0 + X_STEP],   [Y_0],              [Z_0],          [PHI_0],                [THETA_0],                  [PSI_0]])
REFERENCE_TX_N =        np.array([[X_0 - X_STEP],   [Y_0],              [Z_0],          [PHI_0],                [THETA_0],                  [PSI_0]])
REFERENCE_TY =          np.array([[X_0],            [Y_0 + Y_STEP],     [Z_0],          [PHI_0],                [THETA_0],                  [PSI_0]])
REFERENCE_TY_N =        np.array([[X_0],            [Y_0 - Y_STEP],     [Z_0],          [PHI_0],                [THETA_0],                  [PSI_0]])
REFERENCE_TZ =          np.array([[X_0],            [Y_0],              [Z_0 + Z_STEP], [PHI_0],                [THETA_0],                  [PSI_0]])
REFERENCE_TZ_N =        np.array([[X_0],            [Y_0],              [Z_0 - Z_STEP], [PHI_0],                [THETA_0],                  [PSI_0]])
REFERENCE_RX =          np.array([[X_0],            [Y_0],              [Z_0],          [PHI_0 + PHI_STEP],     [THETA_0],                  [PSI_0]])
REFERENCE_RX_N =        np.array([[X_0],            [Y_0],              [Z_0],          [PHI_0 - PHI_STEP],     [THETA_0],                  [PSI_0]])
REFERENCE_RY =          np.array([[X_0],            [Y_0],              [Z_0],          [PHI_0],                [THETA_0 + THETA_STEP],     [PSI_0]])
REFERENCE_RY_N =        np.array([[X_0],            [Y_0],              [Z_0],          [PHI_0],                [THETA_0 - THETA_STEP],     [PSI_0]])
REFERENCE_RZ =          np.array([[X_0],            [Y_0],              [Z_0],          [PHI_0],                [THETA_0],                  [PSI_0 + PHI_STEP]])
REFERENCE_RZ_N =        np.array([[X_0],            [Y_0],              [Z_0],          [PHI_0],                [THETA_0],                  [PSI_0 - PHI_STEP]])
REFERENCE_TXY_PP =      np.array([[X_0 + X_STEP],   [Y_0 + Y_STEP],     [Z_0],          [PHI_0],                [THETA_0],                  [PSI_0]])
REFERENCE_TXY_NN =      np.array([[X_0 - X_STEP],   [Y_0 - Y_STEP],     [Z_0],          [PHI_0],                [THETA_0],                  [PSI_0]])
REFERENCE_TYZ_NP =      np.array([[X_0],            [Y_0 - Y_STEP],     [Z_0 + Z_STEP], [PHI_0],                [THETA_0],                  [PSI_0]])
REFERENCE_RXZ_PN =      np.array([[X_0],            [Y_0],              [Z_0],          [PHI_0 + PHI_STEP],     [THETA_0],                  [PSI_0 - PHI_STEP]])
REFERENCE_TXY_RZ_NPP =  np.array([[X_0 - X_STEP],   [Y_0 + Y_STEP],     [Z_0],          [PHI_0],                [THETA_0],                  [PSI_0 + PHI_STEP]])
REFERENCE_R_LEADER =    np.array([[X_0 - X_STEP/2], [Y_0 + Y_STEP/2],   [Z_0],          [PHI_0],                [THETA_0],                  [PSI_0 + PHI_STEP]])

TASK_REFERENCE_DICTIONARY = {
    "REST":         REFERENCE_REST,
    "TX":           REFERENCE_TX,
    "TX_N":         REFERENCE_TX_N,
    "TY":           REFERENCE_TY,
    "TY_N":         REFERENCE_TY_N,
    "TZ":           REFERENCE_TZ,
    "TZ_N":         REFERENCE_TZ_N,
    "RX":           REFERENCE_RX,
    "RX_N":         REFERENCE_RX_N,
    "RY":           REFERENCE_RY,
    "RY_N":         REFERENCE_RY_N,
    "RZ":           REFERENCE_RZ,
    "RZ_N":         REFERENCE_RZ_N,
    "TXY_PP":       REFERENCE_TXY_PP,
    "TXY_NN":       REFERENCE_TXY_NN,
    "TYZ_NP":       REFERENCE_TYZ_NP,
    "RXZ_PN":       REFERENCE_RXZ_PN,
    "TXY_RZ_NPP":   REFERENCE_TXY_RZ_NPP,
    "R_leader":     REFERENCE_R_LEADER,
}

# The time that marks a team was unable to finish a task
COMPLETED_TIME = 14.75

# Average completion time for each task
TASK_COMPLETION_TIME_DICTIONARY = {
    "REST":         4.5,
    "TX":           4.0,
    "TX_N":         4.25,
    "TY":           5.25,
    "TY_N":         5.5,
    "TZ":           4.75,
    "TZ_N":         4.75,
    "RX":           7.25,
    "RX_N":         6.5,
    "RY":           7.5,
    "RY_N":         8.25,
    "RZ":           5.75,
    "RZ_N":         5.75,
    "TXY_PP":       6.0,
    "TXY_NN":       6.0,
    "TYZ_NP":       7.5,
    "RXZ_PN":       8.75,
    "TXY_RZ_NPP":   8.0,
    "R_leader":     7.0,
}

# Information for looking up paths
GROUPS = [
    '340_341_342', '343_344_345', '352_353_354', '355_356_357', '358_359_360',
    '361_362_363', '364_365_366', '367_368_369', '376_377_378', '379_380_381',
    '382_383_384', '391_392_393', '394_395_396', '400_401_402', '403_404_405'
]

TASKS = [
    "REST",
    "TX","TX_N",
    "TY","TY_N",
    "TZ","TZ_N",
    "RX","RX_N",
    "RY","RY_N",
    "RZ","RZ_N",
    "TXY_PP","TXY_NN",
    "TYZ_NP",
    "RXZ_PN",
    "TXY_RZ_NPP","R_leader"
]

GROUP_TYPES = ['LF', 'LFF', 'LL']

RAW_DATA_FOLDER = "data_raw"
TRAINED_MODELS_FOLDER = "models_trained"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_FOLDER_PATH = os.path.join(CURRENT_DIR, '../..', RAW_DATA_FOLDER)
TRAINED_MODELS_FOLDER_PATH = os.path.join(CURRENT_DIR, '../..', TRAINED_MODELS_FOLDER)
SUBFOLDER = "Fused"

# Rotation matrices
def rot_x(phi):
    """
    Rotation matrix around the X-axis.
    Args:
        phi (float): Rotation angle in radians.
    Returns:
        np.ndarray: Rotation matrix.
    """
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(phi), np.sin(phi)],
        [0.0, -np.sin(phi), np.cos(phi)]
    ])

def rot_y(theta):
    """
    Rotation matrix around the Y-axis.
    Args:
        theta (float): Rotation angle in radians.
    Returns:
        np.ndarray: Rotation matrix.
    """
    return np.array([
        [np.cos(theta), 0.0, -np.sin(theta)],
        [0.0, 1.0, 0.0],
        [np.sin(theta), 0.0, np.cos(theta)]
    ])

def rot_z(psi):
    """
    Rotation matrix around the Z-axis.
    Args:
        psi (float): Rotation angle in radians.
    Returns:
        np.ndarray: Rotation matrix.
    """
    return np.array([
        [np.cos(psi), np.sin(psi), 0.0],
        [-np.sin(psi), np.cos(psi), 0.0],
        [0.0, 0.0, 1.0]
    ])

def rot_zyx(phi, theta, psi):
    """
    Combined rotation matrix for rotations around X, Y, and Z axes.
    Args:
        phi (float): Rotation angle around X-axis in radians.
        theta (float): Rotation angle around Y-axis in radians.
        psi (float): Rotation angle around Z-axis in radians.
    Returns:
        np.ndarray: Combined rotation matrix.
    """
    return rot_x(phi) @ rot_y(theta) @ rot_z(psi)

def table_measured_to_simulation(phi,theta,psi):
    """
    Takes in the euler angles that convert from the inertial frame
    to the measured table frame and converts them to the angles that
    converts from the inertial frame to the simulation table frame.
    The data was converted using the rotations as_euler function which
    returns the data via robotics rotations, but the simulation and
    everything else is done using dynamics rotations.
    
    Args:
        phi (float): Rotation angle around X-axis in radians.
        theta (float): Rotation angle around Y-axis in radians.
        psi (float): Rotation angle around Z-axis in radians.
        
    Returns:
        tuple: Rotation angles (phi_simulation, theta_simulation, psi_simulation) in radians.
    """

    return -phi, -theta, -psi

def table_simulation_euler_to_measured_euler(phi, theta, psi):
    """
    Takes in the Euler angles that convert from the inertial frame
    to the simulation table frame and converts them to the angles that
    convert from the inertial frame to the measured table frame.

    Args:
        phi (float): Rotation angle around X-axis in radians.
        theta (float): Rotation angle around Y-axis in radians.
        psi (float): Rotation angle around Z-axis in radians.

    Returns:
        tuple: Rotation angles (phi_measured, theta_measured, psi_measured) in radians.
    """
    rotation_inertial_to_simulation = rot_zyx(phi, theta, psi)
    rotation_simulation_to_measured = rot_y(-np.pi)
    rotation_inertial_to_measured = rotation_simulation_to_measured @ rotation_inertial_to_simulation
    rotation_obj = Rotation.from_matrix(rotation_inertial_to_measured)
    phi_measured, theta_measured, psi_measured = rotation_obj.as_euler('zyx', degrees=False)
    return phi_measured, theta_measured, psi_measured