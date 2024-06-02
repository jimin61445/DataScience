import pandas as pd
from sklearn.preprocessing import Normalizer
import seaborn as sns
import matplotlib.pyplot as plt

time_columns = ['06시이전', '06-07시간대', '07-08시간대', '08-09시간대', '09-10시간대', '10-11시간대', '11-12시간대', '12-13시간대', '13-14시간대', '14-15시간대', '15-16시간대', '16-17시간대', '17-18시간대', '18-19시간대', '19-20시간대', '20-21시간대', '21-22시간대', '22-23시간대', '23-24시간대', '24시이후']

def scale_csv_standard_row(row):
    mean = row.mean()
    std = row.std()
    return( row -mean )/std


df = pd.read_csv("test_dataset/after_handling_nan.csv",encoding='cp949')
df = df.drop(df.columns[:3], axis=1)

print(df)

df[time_columns] = df[time_columns].apply(scale_csv_standard_row,axis=1)

print(df)

# 스케일된 데이터를 새로운 CSV 파일로 저장
df.to_csv("test_dataset/standard_row.csv", index=False, encoding='cp949')