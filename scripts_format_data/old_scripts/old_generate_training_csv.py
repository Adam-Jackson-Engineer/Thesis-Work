import os
import pandas as pd
import numpy as np

folder = "Data"
groups = ['340_341_342', '343_344_345', '352_353_354', '355_356_357', '358_359_360', '361_362_363', '364_365_366', '367_368_369', '376_377_378', '379_380_381', '382_383_384', '391_392_393', '394_395_396', '400_401_402', '403_404_405']
subFolder = "Fused"
# tasks = ['R_leader', 'RXZ_PN', 'TX', 'RY', 'TY', 'RZ', 'RX_N', 'RZ_N', 'REST', 'RY_N', 'TZ', 'TZ_N', 'TXY_RZ_NPP', 'RX', 'TXY_NN', 'TYZ_NP', 'TY_N', 'TX_N', 'TXY_PP']
# simp_2D = ['TX','TX_N','TY','TY_N','RZ','RZ_N','TXY_NN','TXY_PP']
tasks_2D = ['R_leader','TX','TX_N','TY','TY_N','RZ','RZ_N','TXY_NN','TXY_PP','TXY_RZ_NPP']

# groupTypes = ['LF','LFF','LL']
groupTypes = ['LF','LFF']
abbDataTypes = ['Pos','Vel','Acc','CombFTA_','CombFTB_']

allTimeReq = np.array([5,8,10,12,14])

for timeReq in allTimeReq:
    saveName = f'TrainingData_{timeReq}sec_2D.csv'
    saveFolder = 'trainingData'
    savePath = os.path.join(saveFolder,saveName)

    roundsUsed = 0

    # Ensure the save folder exists
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    if os.path.exists(savePath):
        os.remove(savePath)

    for group in groups:
        for task in tasks_2D:
            path = os.path.join(folder,group,subFolder,task)
            for typ in groupTypes:
                file_start = f'Meta_{group}_{task}_{typ}_'
                for fileName in os.listdir(path):
                    if fileName.startswith(file_start) and fileName.endswith("RPY.csv"):
                        data = pd.read_csv(os.path.join(path,fileName))
                        allTimes = data['Pos_T'].values
                        compTime = data['Pos_T'].iloc[-1] - data['Pos_T'].iloc[0]
                        if compTime < timeReq:
                            roundsUsed += 1
                            data['Group'] = group
                            data['Task'] = task
                            data['Type'] = typ
                            data['CompletionTime'] = compTime
                            data['RoundsUsed'] = roundsUsed
                            if not os.path.exists(savePath):
                                header_columns = ",".join(data.columns) + "\n"
                                with open(savePath, 'w') as f:
                                    f.write(header_columns)

                            data.to_csv(savePath, mode='a', header=False, index=False)
                            print(f'Added {fileName}')

