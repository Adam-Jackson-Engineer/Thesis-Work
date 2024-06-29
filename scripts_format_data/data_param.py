"""
This module contains parameters that define the groupings, tasks, and types
used in processing experimental data for the Learned Follower NN project.

GROUPS: List of strings representing different experimental groups.
TASKS: List of tasks for which data is collected.
GROUP_TYPES: Different types of group categorizations.

Author: Adam Jackson
"""

import os
import numpy as np

GROUPS = [
    '340_341_342', '343_344_345', '352_353_354', '355_356_357', '358_359_360',
    '361_362_363', '364_365_366', '367_368_369', '376_377_378', '379_380_381',
    '382_383_384', '391_392_393', '394_395_396', '400_401_402', '403_404_405'
]

TASKS = [
    'R_leader', 'RXZ_PN', 'TX', 'RY', 'TY', 'RZ', 'RX_N', 'RZ_N', 'REST',
    'RY_N', 'TZ', 'TZ_N', 'TXY_RZ_NPP', 'RX', 'TXY_NN', 'TYZ_NP', 'TY_N',
    'TX_N', 'TXY_PP'
]

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

TASKS_2D = [
    'R_leader','TX','TX_N','TY','TY_N','RZ','RZ_N','TXY_NN','TXY_PP','TXY_RZ_NPP'
    ]

TASKS_2D_1DOF = [
    'TX','TX_N','TY','TY_N','RZ','RZ_N','TXY_NN','TXY_PP'
    ]

LF_GROUPS_TYPE = ['LF']
LFF_GROUPS_TYPE = ['LFF']
FOLLOWER_GROUPS_TYPES = ['LF', 'LFF']
GROUPS_TYPES = ['LF', 'LFF', 'LL']

RAW_DATA_FOLDER = "data_raw"
TRAINING_DATA_FOLDER = "data_training"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_FOLDER_PATH = os.path.join(CURRENT_DIR, '..', RAW_DATA_FOLDER)
SUBFOLDER = "Fused"
TRAINING_DATA_FOLDER_PATH = os.path.join(CURRENT_DIR, '..', TRAINING_DATA_FOLDER)

# The task destination
X_0 = 0.0
Y_0 = 0.0
Z_0 = 0.0
PHI_0 = 0.0
THETA_0 = 0.0
PSI_0 = 0.0       # The table frame is oriented as a 180 deg rotation from the inertial frame

X_STEP = 1.0
Y_STEP = 1.0
Z_STEP = 0.5
PHI_STEP = np.pi / 2
THETA_STEP = np.pi / 4
PSI_STEP = np.pi / 2

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
