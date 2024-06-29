import numpy as np

# Max completion time limit
allTimeReq = np.array([5,8,10,12,14])
TimeReqNames = ['05','08','10','12','14']

# All possible task combinations
trans_x_tasks = ['TX','TX_N']
trans_y_tasks = ['TY','TY_N']
trans_xy_tasks = ['TX','TX_N','TY','TY_N','TXY_NN','TXY_PP']
rot_z_tasks = ['RZ','RZ_N']
trans_rot_tasks = ['R_leader','TXY_RZ_NPP']
tasks_2D = ['R_leader','TX','TX_N','TY','TY_N','RZ','RZ_N','TXY_NN','TXY_PP','TXY_RZ_NPP']

allTasksCombo = [trans_x_tasks, trans_y_tasks, trans_xy_tasks, rot_z_tasks, trans_rot_tasks, tasks_2D]
TaskComboNames = ['TX', 'TY', 'XY', 'RZ', 'TR','2D']

# All possible time lengths
srt = 0.25
crt = 0.75
sampling_rate = 200
SRT_steps = sampling_rate * srt
CRT_steps = sampling_rate * crt

allRTCombo = [SRT_steps, CRT_steps]
RTComboNames = ['SRT', 'CRT']

# All input options
vel_only_inputs = ['Vel_X', 'Vel_Y', 'Vel_Psi']
vel_acc_inputs = ['Vel_X', 'Vel_Y', 'Vel_Psi', 'Acc_X', 'Acc_Y', 'Acc_Psi']
tao_inputs = ['CombFTA__X', 'CombFTA__Y', 'CombFTA__Psi']
vel_acc_tao_inputs = ['Vel_X', 'Vel_Y', 'Vel_Psi', 'Acc_X', 'Acc_Y', 'Acc_Psi', 'CombFTA__X', 'CombFTA__Y', 'CombFTA__Psi']

allInputCombo = [vel_only_inputs, vel_acc_inputs, tao_inputs, vel_acc_tao_inputs]
InputComboNames = ['vel','vac','tao','vat']

# All possible output options
tao_x_output = ['CombFTB__X']
tao_y_output = ['CombFTB__Y']
tao_psi_output = ['CombFTB__Psi']
tao_xy_output = ['CombFTB__X','CombFTB__Y']
tao_xypsi_output = ['CombFTB__X', 'CombFTB__Y', 'CombFTB__Psi']

allOutputCombo = [tao_x_output, tao_y_output, tao_psi_output, tao_xy_output, tao_xypsi_output]
OutputComboNames = ['XXX','YYY','PSI','tXY','XYS']

