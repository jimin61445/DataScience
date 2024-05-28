import pandas as pd
import numpy as np

df_Nop = pd.read_csv('test_dataset/categorical_dataset.csv',encoding='cp949')

time_column = ['06시이전','06-07시간대','07-08시간대','08-09시간대','09-10시간대','10-11시간대','11-12시간대','12-13시간대','13-14시간대','14-15시간대','15-16시간대','16-17시간대','17-18시간대','18-19시간대','19-20시간대','20-21시간대','21-22시간대','22-23시간대','23-24시간대','24시이후']

second_time_column =['07-08시간대','08-09시간대','09-10시간대','10-11시간대','11-12시간대','12-13시간대','13-14시간대','14-15시간대','15-16시간대','16-17시간대','17-18시간대','18-19시간대','19-20시간대','20-21시간대','21-22시간대','22-23시간대','23-24시간대']

date_threshold = pd.Timestamp('2023-06-30')
df_Nop['수송일자'] = pd.to_datetime(df_Nop["수송일자"])

# 7월 1일 데이터 이후 '06-07시간대' 0으로 데이터가 전부 이상하여 0으로 고정하여 결측치로 판단
df_Nop.loc[df_Nop['수송일자'] > date_threshold, '06-07시간대'] = 0

# 0을 np.nan으로 변환
df_Nop[time_column] = df_Nop[time_column].replace(0,np.nan)

nan_count = df_Nop.isnull().sum()

print(nan_count)


# 시간대 열에서 모든 데이터가 결측치 인 경우 삭제 
df_Nop = df_Nop.dropna(how='all',subset=time_column)

nan_count = df_Nop.isnull().sum()

print(nan_count)

filter_data = df_Nop[df_Nop['수송일자']<=date_threshold]
target_data = df_Nop[df_Nop['수송일자']>date_threshold]
mean_until_july = filter_data.groupby(['역번호','승하차구분']).mean()
mean = df_Nop.groupby(['역번호','승하차구분']).mean()

print(mean_until_july)
mean_until_july.to_csv('test_dataset/mean_until_july.csv',encoding='cp949')

def fill_from_july(row):
    station = row['역번호']
    ride = row['승하차구분']
    col = '06-07시간대'  # Specify the column to fill

    if pd.isna(row[col]):
        row[col] = mean_until_july.loc[(station, ride), col]

    return row

def fill_all(row):
    station = row['역번호']
    ride = row['승하차구분']
    
    for col in row.index:
        if pd.isna(row[col]):
            row[col]=mean.loc[(station,ride),col]

    return row


# 7월 1일 이후 데이터에 06-07시간대 데이터 이상으로 6월 30일 이전 데이터로 평균지어 결측치 채움
target_data = target_data.apply(fill_from_july, axis=1)

df_Nop.loc[df_Nop['수송일자'] > date_threshold, '06-07시간대'] = target_data['06-07시간대']

# 첫차 시간대와 막차 시간대를 제외한 시간의 결측치를 평균으로 대체
second_target_data =  df_Nop.apply(fill_all,axis=1)

df_Nop[second_time_column] = second_target_data[second_time_column]

print(df_Nop)

# df_Nop[time_column] = df_Nop[time_column].replace(np.nan,0)

nan_count = df_Nop.isnull().sum()

print(nan_count)


# 첫차와 막차를 제외한 시간대에서 결측치가 하나라도 존재하는 경우 즉 데이터가 모두 결측치이거나 데이터가 소수만 존재하는 경우 삭제
df_Nop = df_Nop.dropna(how='any',subset=second_time_column)

nan_count = df_Nop.isnull().sum()

print(nan_count)


# 6시 이전 운행안하는 경우와 24시이후 운행안하는 경우 고려하여 나머지 결측치들은 0으로 대체
df_Nop[time_column] = df_Nop[time_column].replace(np.nan,0)


df_Nop.to_csv("test_dataset/after_handling_nan.csv",encoding='cp949')
