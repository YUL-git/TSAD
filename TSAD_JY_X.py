import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  ## MULTI GPU 사용시 고정

#----------------------------------hyperparameter setting--------------------------------------#
from argparse import ArgumentParser
parser = ArgumentParser(description="Univariate-Exogenous TSAD")

parser.add_argument('--train_path', default='data/raw_data/train/Training_Data.csv', type=str, help='path to the data')
parser.add_argument('--test1_path', default='data/raw_data/test/WeldingTest_01_OK.csv', type=str, help='path to the data')
parser.add_argument('--test2_path', default='data/raw_data/test/WeldingTest_02_OK.csv', type=str, help='path to the data')
parser.add_argument('--test3_path', default='data/raw_data/test/WeldingTest_03_NG.csv', type=str, help='path to the data')
parser.add_argument('--test4_path', default='data/raw_data/test/WeldingTest_04_NG.csv', type=str, help='path to the data')
parser.add_argument('--label3_path', default='data/preprocessed/test/WeldingTest_03_NG_Label.csv', type=str, help='path to the data')

parser.add_argument('--horizon', default=10, type=int, help='forecast horizon')
parser.add_argument('--input_size', default=20, type=int, help='input size')
parser.add_argument('--threshold', default=4.0, type=float, help='threshold')

args = parser.parse_args()

CFG = {
    "train"  : args.train_path,
    "test1"  : args.test1_path,
    "test2"  : args.test2_path,
    "test3"  : args.test3_path,
    "test4"  : args.test4_path,
    "label3" : args.label3_path,
    
    "horizon"     : args.horizon,
    "input_size"  : args.input_size,
    "threshold"   : args.threshold
    }
#----------------------------------------------------------------------------------------------#

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

root = os.getcwd()
path = os.path.join(root, CFG['train']) # path 수정
data = pd.read_csv(path)
data = data.rename(columns=lambda col_name: col_name.strip())

data.drop(columns=['SetFrequency', 'SetDuty'], inplace=True)

cols = [data.columns[3]]
df_rp = data[cols]
for i in range(len(data)):
    if data[data.columns[3]][i] < 1300.0:
        df_rp[data.columns[3]][i] = df_rp[data.columns[3]][i] +930.5

data.loc[:, 'RealPower'] = df_rp

use_columns = data.columns.tolist()

data = data[:10800]

path  = os.path.join(root, CFG['test1']) # path 수정
test1 = pd.read_csv(path)
test1 = test1.rename(columns=lambda col_name: col_name.strip())
test1 = test1[use_columns]

cols = [test1.columns[3]]

df_test1 = test1[cols]
for i in range(len(test1)):
    if test1[test1.columns[3]][i] <1300.0:
        df_test1[test1.columns[3]][i] = df_test1[test1.columns[3]][i] +930.5

test1.loc[:, 'RealPower'] = df_test1

path  = os.path.join(root, CFG['test2']) # path 수정
test2 = pd.read_csv(path)
test2 = test2.rename(columns=lambda col_name: col_name.strip())
test2 = test2[use_columns]

cols = [test2.columns[3]]

df_test2 = test2[cols]
for i in range(len(test2)):
    if test2[test2.columns[3]][i] <1300.0:
        df_test2[test2.columns[3]][i] = df_test2[test2.columns[3]][i] +930.5

test2.loc[:, 'RealPower'] = df_test2

path  = os.path.join(root, CFG['test3']) # path 수정
test3 = pd.read_csv(path)
test3 = test3.rename(columns=lambda col_name: col_name.strip())
test3 = test3[use_columns]

cols = [test3.columns[3]]

df_test3 = test3[cols]
for i in range(len(test3)):
    if test3[test3.columns[3]][i] <1300.0:
        df_test3[test3.columns[3]][i] = df_test3[test3.columns[3]][i] +930.5

test3.loc[:, 'RealPower'] = df_test3

path  = os.path.join(root, CFG['test4']) # path 수정
test4 = pd.read_csv(path)
test4 = test4.rename(columns=lambda col_name: col_name.strip())
test4 = test4[use_columns]

cols = [test4.columns[3]]

df_test4 = test4[cols]
for i in range(len(test4)):
    if test4[test4.columns[3]][i] <1300.0:
        df_test4[test4.columns[3]][i] = df_test4[test4.columns[3]][i] +930.5

test4.loc[:, 'RealPower'] = df_test4

unique_id_cols = ['PageNo', 'Speed', 'Length', 'RealPower', 'SetPower', 'GateOnTime']

data['ds'] = pd.to_datetime(data.index, unit='s')

train_data = data.rename(columns={'RealPower': 'y'}).assign(
    unique_id='RealPower'
)

train_data  = train_data[['ds', 'unique_id', 'y'] + [col for col in unique_id_cols if col != 'RealPower']]

test1['ds'] = pd.to_datetime(test1.index, unit='s')
test_data1  = test1.rename(columns={'RealPower': 'y'}).assign(
    unique_id='RealPower'
)
test_data1  = test_data1[['ds', 'unique_id', 'y'] + [col for col in unique_id_cols if col != 'RealPower']]

test2['ds'] = pd.to_datetime(test2.index, unit='s')
test_data2  = test2.rename(columns={'RealPower': 'y'}).assign(
    unique_id='RealPower'
)
test_data2  = test_data2[['ds', 'unique_id', 'y'] + [col for col in unique_id_cols if col != 'RealPower']]

test3['ds'] = pd.to_datetime(test3.index, unit='s')
test_data3  = test3.rename(columns={'RealPower': 'y'}).assign(
    unique_id='RealPower'
)
test_data3  = test_data3[['ds', 'unique_id', 'y'] + [col for col in unique_id_cols if col != 'RealPower']]

test4['ds'] = pd.to_datetime(test4.index, unit='s')
test_data4  = test4.rename(columns={'RealPower': 'y'}).assign(
    unique_id='RealPower'
)
test_data4  = test_data4[['ds', 'unique_id', 'y'] + [col for col in unique_id_cols if col != 'RealPower']]


from neuralforecast import NeuralForecast
from neuralforecast.models import TSMixerx, NHITS, NBEATSx, LSTM, GRU
from neuralforecast.losses.pytorch import MAE, MQLoss, MSE

import math
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

horizon    = CFG['horizon']
input_size = CFG['input_size']

models = [
            TSMixerx(
                h           = horizon,
                input_size  = input_size,
                n_series    = 1,            ## exogenous 변수가 없기 때문 (외부 변수)
                scaler_type = 'identity',
                max_steps   = 500,
                loss        = MSE(),
                valid_loss  = MSE(),
                batch_size  = 128,
                random_seed = 42,
                val_check_steps = 100,
                early_stop_patience_steps = 5,
                hist_exog_list = ['PageNo', 'Speed', 'Length', 'SetPower', 'GateOnTime'],
                ),

            NHITS(
                h           = horizon,
                input_size  = input_size,
                scaler_type = 'identity',
                loss        = MSE(),
                valid_loss  = MSE(),
                batch_size  = 128,
                random_seed = 42,
                val_check_steps = 5,
                hist_exog_list = ['PageNo', 'Speed', 'Length', 'SetPower', 'GateOnTime'],
                early_stop_patience_steps = -1,
                ),

            NBEATSx(
                h           = horizon,
                input_size  = input_size,
                scaler_type = 'identity',
                loss        = MSE(),
                valid_loss  = MSE(),
                batch_size  = 128,
                random_seed = 42,
                val_check_steps = 5,
                hist_exog_list = ['PageNo', 'Speed', 'Length', 'SetPower', 'GateOnTime'],
                early_stop_patience_steps = -1,
                ),
            
            LSTM(
                h           = horizon,
                input_size  = input_size,
                scaler_type = 'identity',
                loss        = MSE(),
                valid_loss  = MSE(),
                batch_size  = 128,
                random_seed = 42,
                val_check_steps = 5,
                hist_exog_list = ['PageNo', 'Speed', 'Length', 'SetPower', 'GateOnTime'],
                early_stop_patience_steps = -1,
                ),
            
            GRU(
                h           = horizon,
                input_size  = input_size,
                scaler_type = 'identity',
                loss        = MSE(),
                valid_loss  = MSE(),
                batch_size  = 128,
                random_seed = 42,
                val_check_steps = 5,
                hist_exog_list = ['PageNo', 'Speed', 'Length', 'SetPower', 'GateOnTime'],
                early_stop_patience_steps = -1,
                ),
            
            ]

nf = NeuralForecast(models = models, freq = 's')
train_hat_data = nf.cross_validation(
                                    df = train_data,
                                    val_size  = CFG['horizon'],
                                    test_size = CFG['horizon'],
                                    n_windows = None,
                                    verbose   = False)

train_hat_data = train_hat_data.reset_index()

## 예측을 위해 windowed dataset 생성 (NIXTLA 맞춤)

def create_windowed_dataset(data, window_length=input_size, stride=horizon):
    
    total_windowed_data = []
    total_y_true_data   = []

    unique_ids = data['unique_id'].unique()
    sample_data = data[data['unique_id'] == unique_ids[0]]
    
    for start in range(0, len(sample_data) - window_length - horizon + 1, stride):
        windowed_data = []
        y_true_data   = []
    
        end = start + window_length

        for unique_id in unique_ids:
            data_id = data[data['unique_id'] == unique_id]
            window = data_id.iloc[start:end]
            windowed_data.append(window)
            
            y_true = data_id.iloc[end:end+horizon]
            y_true_data.append(y_true)

        total_windowed_data.append(pd.concat(windowed_data))
        total_y_true_data.append(pd.concat(y_true_data))
        
    return total_windowed_data, total_y_true_data

## windowed dataset 예측

def time_series_anomaly_detection(nf, total_windowed_data):
    
    forecasts = []
    
    for window in total_windowed_data:

        forecast = nf.predict(window, verbose=False)
        forecasts.append(forecast)

    return forecasts

def anomaly_score(merged_df, gt, error_name_list, thres=CFG['threshold']):

    total_length = merged_df.shape[0]
    gt   = gt[20 : 20 + total_length]
    gt   = np.array(gt)
    threshold = thres

    for model in model_name_list:

        final_scores = merged_df[f'{model}_error'].to_numpy()
        avg   = np.mean(final_scores)
        sigma = np.std(final_scores)

        Z_score = (final_scores - avg) / sigma
        pred = (np.abs(Z_score) > threshold).astype(int)
        pred = np.array(pred)
        
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
            average='binary')
        
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(
            accuracy, precision,
            recall, f_score), f" --- {model}")

total_windowed_data, total_y_true_data = create_windowed_dataset(test_data3)

forecasts = time_series_anomaly_detection(nf, total_windowed_data)

forecasts_df = pd.concat(forecasts)
forecasts_df = forecasts_df.sort_values(by=["unique_id", "ds"])
forecasts_df['unique_id'] = forecasts_df.index
forecasts_df.reset_index(drop=True, inplace=True)

total_y_true_df = pd.concat(total_y_true_data)
total_y_true_df = total_y_true_df.sort_values(by=["unique_id", "ds"])

merged_df = pd.merge(total_y_true_df, forecasts_df, on=["ds", "unique_id"], how="inner")
for model in models:
    merged_df[f'{model}_error'] = merged_df[f'{model}'] - merged_df['y']

model_name_list = ['TSMixerx', 'NHITS', 'NBEATSx', 'LSTM', 'GRU']
realpower_df  = merged_df[merged_df["unique_id"] == "RealPower"]
label_df      = pd.read_csv(os.path.join(root, CFG['label3']))
true_label    = label_df['label']

anomaly_score(realpower_df, true_label, model_name_list)