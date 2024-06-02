import pandas as pd 
import numpy as np 

df_Nop = pd.read_csv("original_dataset/dataset_Nop.csv",encoding="cp949")
df_Congestion = pd.read_csv("original_dataset/dataset_Congestion.csv",encoding="cp949")

time_column = ["5시30분","6시00분","6시30분","7시00분","7시30분","8시00분","8시30분","9시00분","9시30분","10시00분","10시30분","11시00분","11시30분","12시00분","12시30분","13시00분","13시30분","14시00분","14시30분","15시00분","15시30분","16시00분","16시30분","17시00분","17시30분","18시00분","18시30분","19시00분","19시30분","20시00분","20시30분","21시00분","21시30분","22시00분","22시30분","23시00분","23시30분","00시00분","00시30분"]

print(df_Nop.head())
print(df_Congestion.head())

print(df_Nop.info())
print(df_Congestion.info())

df_Congestion[time_column] = df_Congestion[time_column].replace(0,np.nan)

# 시간대 열에서 모든 데이터가 결측치 인 경우 삭제 
df_Congestion = df_Congestion.dropna(how='all',subset=time_column)

row_means = df_Congestion.mean(axis=0)
df_Congestion["22시30분"] = df_Congestion["22시30분"].fillna(value=row_means,axis=0)


print(df_Congestion)


df_Congestion.to_csv("test_dataset/congestion_nan.csv",encoding='cp949')

nan_count = (df_Nop == 0).sum()

print(nan_count)

nan_count = df_Congestion.isnull().sum()

print(nan_count)
