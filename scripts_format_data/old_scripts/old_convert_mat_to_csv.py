import os
import time
import scipy.io
import numpy as np

folder = "Data"
groups = ['340_341_342', '343_344_345', '352_353_354', '355_356_357', '358_359_360', '361_362_363', '364_365_366', '367_368_369', '376_377_378', '379_380_381', '382_383_384', '391_392_393', '394_395_396', '400_401_402', '403_404_405']
subFolder = "Fused"
tasks = ['R_leader', 'RXZ_PN', 'TX', 'RY', 'TY', 'RZ', 'RX_N', 'RZ_N', 'REST', 'RY_N', 'TZ', 'TZ_N', 'TXY_RZ_NPP', 'RX', 'TXY_NN', 'TYZ_NP', 'TY_N', 'TX_N', 'TXY_PP']
# tasks_2D = ['R_leader','TX','TX_N','TY','TY_N','RZ','RZ_N','TXY_NN','TXY_PP','TXY_RZ_NPP']
# simp_2D = ['TX','TX_N','TY','TY_N','RZ','RZ_N','TXY_NN','TXY_PP']
groupTypes = ['LF','LFF','LL']
# dataTypes = ['AccelTraj','PoseTraj','VelTraj','ft1','ft2','ft3','ft4']
# abbDataTypes = ['Pos','Vel','Acc','ft1','ft2','ft3','ft4']
abbDataTypes = ['Pos','Vel','Acc','CombFTA_','CombFTB_']
kinematic = ['T','X','Y','Z','W','I','J','K']
kinetic = ['T','X','Y','Z','Phi','Tht','Psi']

fullLabels = []
for typ in abbDataTypes:
    if typ == abbDataTypes[0] or typ == abbDataTypes[1] or typ == abbDataTypes[2]:
        for kin in kinematic:
            fullLabels.append(f'{typ}_{kin}')
    else:
        for kin in kinetic:
            fullLabels.append(f'{typ}_{kin}')

for group in groups:
    for task in tasks:
        path = os.path.join(folder,group,subFolder,task)
        for typ in groupTypes:
            file_start = f'Meta_{group}_{task}_{typ}_'
            for fileName in os.listdir(path):
                if fileName.startswith(file_start) and fileName.endswith(".mat"):
                    matData = scipy.io.loadmat(os.path.join(path,fileName))
                    arrays = []
                    for dataType in abbDataTypes:
                        variables = [v for v in matData if v.startswith(dataType)]
                        for var in variables:
                            if matData[var].size > 0:
                                arrays.append(matData[var])

                    if arrays:
                        min_length = min(array.shape[1] for array in arrays)
                        preprocessed_arrays = [array[:, :min_length, ...] for array in arrays]
                        allData = np.vstack(preprocessed_arrays)
                        allDataTransposed = allData.T
                        save_path = os.path.join(path,fileName.replace(".mat",".csv"))

                        np.savetxt(save_path, allDataTransposed, delimiter=",", header=",".join(fullLabels),comments='')
                        print(f"Saved combined data to {save_path}")
