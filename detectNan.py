import pandas as pd
import numpy as np

# CSV 파일 읽기
df_Nop = pd.read_csv('test_dataset/categorical_dataset.csv',encoding='cp949')

time_column = ['06시이전','06-07시간대','07-08시간대','08-09시간대','09-10시간대','10-11시간대','11-12시간대','12-13시간대','13-14시간대','14-15시간대','15-16시간대','16-17시간대','17-18시간대','18-19시간대','19-20시간대','20-21시간대','21-22시간대','22-23시간대','23-24시간대','24시이후']

# 0을 np.nan으로 변환
df_Nop[time_column] = df_Nop[time_column].replace(0,np.nan)

# 시간대 열에서 모든 데이터가 결측치 인 경우 삭제 
df_Nop = df_Nop.dropna(how='all',subset=time_column)
print(df_Nop)

# df_Nop[time_column] = df_Nop[time_column].replace(np.nan,0)

# 시간대별로 결측치가 몇개 존재하는지 확인 코드
nan_count = df_Nop.isnull().sum()

print(nan_count)

nan_df = df_Nop[df_Nop[time_column].isna().any(axis=1)]

print(nan_df)

nan_df.to_csv("test_dataset/nan_datset_Nop.csv",encoding='cp949')


# df_Nop['수송일자'] = pd.to_datetime(df_Nop["수송일자"])
# df_Nop['요일구분'] = df_Nop['수송일자'].dt.day_name()


# def change_day(day):
#     if day == 'Sunday':
#         return '일요일'
#     elif day == 'Saturday':
#         return '토요일'
#     else:
#         return '평일'
    
# df_Nop['요일구분'] = df_Nop['요일구분'].apply(change_day)


# new_df_Nop = df_Nop.groupby(['역명','호선','승하차구분','요일구분']).mean(numeric_only=True)
# new_df_Nop.drop(['연번'],axis=1,inplace=True)


# new_df_Nop.to_csv("test_dataset/new_df_Nop2.csv",encoding="cp949")