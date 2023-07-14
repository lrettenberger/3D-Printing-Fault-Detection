import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm

# additional test data




img_path = '/home/ws/kg2371/Promotion/2023/Nils_Beyer_MA_Paper/PRINTING_ERRORS/images/all_images256'
data_base_path = '/home/ws/kg2371/projects/3D-Printing-Fault-Detection/data'
csv_base_path = '/home/ws/kg2371/Promotion/2023/Nils_Beyer_MA_Paper/PRINTING_ERRORS/kfold_data/Geometry_Split_k-fold_train_val_splits/'

for test_fold in tqdm([0,1,2,3,4]):
    for val_fold in [0,1,2,3,4]:
        Path(f'{data_base_path}/test_{test_fold}/{val_fold}/train/0').mkdir(parents=True,exist_ok=True)
        Path(f'{data_base_path}/test_{test_fold}/{val_fold}/train/1').mkdir(parents=True,exist_ok=True)
        Path(f'{data_base_path}/test_{test_fold}/{val_fold}/train/2').mkdir(parents=True,exist_ok=True)
        Path(f'{data_base_path}/test_{test_fold}/{val_fold}/test/0').mkdir(parents=True,exist_ok=True)
        Path(f'{data_base_path}/test_{test_fold}/{val_fold}/test/1').mkdir(parents=True,exist_ok=True)
        Path(f'{data_base_path}/test_{test_fold}/{val_fold}/test/2').mkdir(parents=True,exist_ok=True)
        Path(f'{data_base_path}/test_{test_fold}/{val_fold}/val/0').mkdir(parents=True,exist_ok=True)
        Path(f'{data_base_path}/test_{test_fold}/{val_fold}/val/1').mkdir(parents=True,exist_ok=True)
        Path(f'{data_base_path}/test_{test_fold}/{val_fold}/val/2').mkdir(parents=True,exist_ok=True)

        train_data = pd.read_csv(f'{csv_base_path}/train_fold{test_fold}_fold{val_fold}.csv',sep=';')
        val_data = pd.read_csv(f'{csv_base_path}/val_fold{test_fold}_fold{val_fold}.csv',sep=';')
        test_data = pd.read_csv(f'{csv_base_path}/test_fold{test_fold}.csv',sep=';')
        
        for index, row in train_data.iterrows():
            shutil.copy(f'{img_path}/{row["image"]}', f'{data_base_path}/test_{test_fold}/{val_fold}/train/{row["class"]}/')
        for index, row in val_data.iterrows():
            shutil.copy(f'{img_path}/{row["image"]}', f'{data_base_path}/test_{test_fold}/{val_fold}/val/{row["class"]}/')
        for index, row in test_data.iterrows():
            shutil.copy(f'{img_path}/{row["image"]}', f'{data_base_path}/test_{test_fold}/{val_fold}/test/{row["class"]}/')
            

silberbed_csv = '/home/ws/kg2371/Promotion/2023/Nils_Beyer_MA_Paper/PRINTING_ERRORS/test_data/silver_test.csv'
test_data = pd.read_csv(silberbed_csv,sep=';')
Path(f'{data_base_path}/silver_test/0').mkdir(parents=True,exist_ok=True)
Path(f'{data_base_path}/silver_test/1').mkdir(parents=True,exist_ok=True)
Path(f'{data_base_path}/silver_test/2').mkdir(parents=True,exist_ok=True)
for index, row in test_data.iterrows():
    shutil.copy(f'{img_path}/{row["image"]}', f'{data_base_path}/silver_test/{row["class"]}/')

oblique_csv = '/home/ws/kg2371/Promotion/2023/Nils_Beyer_MA_Paper/PRINTING_ERRORS/test_data/perspective_test.csv'
test_data = pd.read_csv(oblique_csv,sep=';')
Path(f'{data_base_path}/oblique_test/0').mkdir(parents=True,exist_ok=True)
Path(f'{data_base_path}/oblique_test/1').mkdir(parents=True,exist_ok=True)
Path(f'{data_base_path}/oblique_test/2').mkdir(parents=True,exist_ok=True)
for index, row in test_data.iterrows():
    shutil.copy(f'{img_path}/{row["image"]}', f'{data_base_path}/oblique_test/{row["class"]}/')