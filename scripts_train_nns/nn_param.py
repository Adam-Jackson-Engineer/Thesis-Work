"""
nn_param.py

Contains constants and parameters for neural network training.
"""

import numpy as np
import os

# Max completion time limit
CUTOFF_PERC = np.array([0, 25, 50, 75, 100])
CUTOFF_NAMES = ['0', '25', '50', '75', '100']

# All possible task combinations
TRANS_X_TASKS = ['TX', 'TX_N']
TRANS_Y_TASKS = ['TY', 'TY_N']
TRANS_XY_TASKS = ['TX', 'TX_N', 'TY', 'TY_N', 'TXY_NN', 'TXY_PP']
ROT_Z_TASKS = ['RZ', 'RZ_N']
TRANS_ROT_TASKS = ['R_leader', 'TXY_RZ_NPP']
TASKS_2D = ['R_leader', 'TX', 'TX_N', 'TY', 'TY_N', 'RZ', 'RZ_N', 'TXY_NN', 'TXY_PP', 'TXY_RZ_NPP']

ALL_TASKS_COMBO = [TRANS_X_TASKS, TRANS_Y_TASKS, TRANS_XY_TASKS, ROT_Z_TASKS, TRANS_ROT_TASKS, TASKS_2D]
TASK_COMBO_NAMES = ['TX', 'TY', 'XY', 'RZ', 'TR', '2D']

# All possible time lengths
SRT = 0.25
CRT = 0.75
SAMPLING_RATE = 200
SRT_STEPS = int(SAMPLING_RATE * SRT)
CRT_STEPS = int(SAMPLING_RATE * CRT)

ALL_RATE_COMBO = [SRT_STEPS, CRT_STEPS]
RT_COMBO_NAMES = ['SRT', 'CRT']

# All input options
POS_ONLY_INPUTS = ['Pos_X', 'Pos_Y', 'Pos_Psi']
VEL_ONLY_INPUTS = ['Vel_X', 'Vel_Y', 'Vel_Psi']
ACC_ONLY_INPUTS = ['Acc_X', 'Acc_Y', 'Acc_Psi']
TAO_ONLY_INPUTS = ['CombFTA__X', 'CombFTA__Y', 'CombFTA__Psi']

VEL_ACC_TAO_INPUTS = ['Vel_X', 'Vel_Y', 'Vel_Psi', 'Acc_X', 'Acc_Y', 'Acc_Psi', 'CombFTA__X', 'CombFTA__Y', 'CombFTA__Psi']

POS_VEL_INPUTS = ['Pos_X', 'Pos_Y', 'Pos_Psi', 'Vel_X', 'Vel_Y', 'Vel_Psi']
POS_VEL_ACC_INPUTS = ['Pos_X', 'Pos_Y', 'Pos_Psi', 'Vel_X', 'Vel_Y', 'Vel_Psi', 'Acc_X', 'Acc_Y', 'Acc_Psi']
POS_VEL_ACC_TAO_INPUTS = ['Pos_X', 'Pos_Y', 'Pos_Psi', 'Vel_X', 'Vel_Y', 'Vel_Psi', 'Acc_X', 'Acc_Y', 'Acc_Psi', 'CombFTA__X', 'CombFTA__Y', 'CombFTA__Psi']

POS_TAO_INPUTS = ['Pos_X', 'Pos_Y', 'Pos_Psi', 'CombFTA__X', 'CombFTA__Y', 'CombFTA__Psi']

ALL_INPUT_COMBO = [POS_ONLY_INPUTS, VEL_ONLY_INPUTS, ACC_ONLY_INPUTS, TAO_ONLY_INPUTS, VEL_ACC_TAO_INPUTS, POS_VEL_INPUTS, 
                   POS_VEL_ACC_INPUTS, POS_VEL_ACC_TAO_INPUTS,POS_TAO_INPUTS]
INPUT_COMBO_NAMES = ['pos', 'vel', 'acc', 'tao', 'vat', 'pv', 'pva', 'all', 'pt']

# All possible output options
TAO_X_OUTPUT = ['CombFTB__X']
TAO_Y_OUTPUT = ['CombFTB__Y']
TAO_PSI_OUTPUT = ['CombFTB__Psi']
TAO_XY_OUTPUT = ['CombFTB__X', 'CombFTB__Y']
TAO_XYPSI_OUTPUT = ['CombFTB__X', 'CombFTB__Y', 'CombFTB__Psi']
TASK_XYZ_OUTPUT = ['Task_X', 'Task_Y', 'Task_Psi']

ALL_OUTPUT_COMBO = [TAO_X_OUTPUT, TAO_Y_OUTPUT, TAO_PSI_OUTPUT, TAO_XY_OUTPUT, TAO_XYPSI_OUTPUT, TASK_XYZ_OUTPUT]
OUTPUT_COMBO_NAMES = ['X', 'Y', 'PSI', 'XY', 'XYS', 'task']

# Directories
TRAINING_DATA_FOLDER = "data_training"
TRAINED_MODEL_FOLDER = "models_trained"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_FOLDER_PATH = os.path.join(CURRENT_DIR, '..', TRAINING_DATA_FOLDER)
TRAINED_MODEL_FOLDER_PATH = os.path.join(CURRENT_DIR, '..', TRAINED_MODEL_FOLDER)
TRAINED_MODEL_FOLDER_PRIMARY = "models_primary"
TRAINED_MODEL_FOLDER_ALL = "models_all"
TRAINED_MODEL_FOLDER_PRIMARY_PATH = os.path.join(TRAINED_MODEL_FOLDER_PATH, TRAINED_MODEL_FOLDER_PRIMARY)
TRAINED_MODEL_FOLDER_ALL_PATH = os.path.join(TRAINED_MODEL_FOLDER_PATH, TRAINED_MODEL_FOLDER_ALL)


def generate_model_list():
    """
    Generates a list of all model names based on the combinations of parameters.
    
    Returns:
        list: A list of all model names.
    """
    model_list = []
    for i, cutoff_time in enumerate(CUTOFF_PERC):
        for j, tasks in enumerate(ALL_TASKS_COMBO):
            for k, steps in enumerate(ALL_RATE_COMBO):
                for m, input_columns in enumerate(ALL_INPUT_COMBO):
                    for n, output_columns in enumerate(ALL_OUTPUT_COMBO):
                        model_name = (
                            RT_COMBO_NAMES[k] + "_" +
                            OUTPUT_COMBO_NAMES[n] + "_" +
                            INPUT_COMBO_NAMES[m] + "_" +
                            TASK_COMBO_NAMES[j] + "_" +
                            CUTOFF_NAMES[i]
                        )
                        model_list.append(model_name)
    return model_list

def generate_primary_model_list(PREDICTED_BEST):
    """
    Generates a list of the primary model names where only 1 parameter
    changes from the primary list at a time.
    
    Returns:
        list: A list of primary model names.
    """
    model_list = []

    for i, rate in enumerate(ALL_RATE_COMBO):
        model_name = replace_one_element(PREDICTED_BEST,0,RT_COMBO_NAMES[i])
        model_list.append(model_name)

    for i, output_combo in enumerate(ALL_OUTPUT_COMBO):
        model_name = replace_one_element(PREDICTED_BEST,1,OUTPUT_COMBO_NAMES[i])
        model_list.append(model_name)

    for i, input_combo in enumerate(ALL_INPUT_COMBO):
        model_name = replace_one_element(PREDICTED_BEST,2,INPUT_COMBO_NAMES[i])
        model_list.append(model_name)

    for i, task in enumerate(TASK_COMBO_NAMES):
        model_name = replace_one_element(PREDICTED_BEST,3,TASK_COMBO_NAMES[i])
        model_list.append(model_name)

    for i, cutoff_perc in enumerate(CUTOFF_PERC):
        model_name = replace_one_element(PREDICTED_BEST,4,CUTOFF_NAMES[i])
        model_list.append(model_name)

    return model_list

def replace_one_element(model_name,element_index,new_element):
    """
    Takes in the primary model name, the index to change and the new element and 
    creates a new model name
    
    Args:
    model_name (str): The name of the model to modify
    element_index (int): The index of element in the name to modify
    new_element (str): The new element to add to the model name
    
    Returns:
    new_model_name (str): The modified model name
    """
    model_name_parts = model_name.split("_")
    model_name_parts[element_index] = new_element
    new_model_name = "_".join(model_name_parts)

    return new_model_name

MODEL_ALL_LIST = generate_model_list()

# Model names follow: {reaction_time}_{output}_{input}_{task_type}_{percentile}

PREDICTED_BEST = 'SRT_XY_all_2D_75'

MODEL_PRIMARY_LIST = generate_primary_model_list(PREDICTED_BEST)
