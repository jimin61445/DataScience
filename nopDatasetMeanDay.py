import pandas as pd 
import numpy as np 

df_Nop = pd.read_csv("original_dataset/dataset_Nop.csv",encoding="cp949")
df_Congestion = pd.read_csv("original_dataset/dataset_Congestion.csv",encoding="cp949")

print(df_Nop.head())
print(df_Congestion.head())

df_Nop['수송일자'] = pd.to_datetime(df_Nop["수송일자"])
df_Nop['요일구분'] = df_Nop['수송일자'].dt.day_name()


def change_day(day):
    if day == 'Sunday':
        return '일요일'
    elif day == 'Saturday':
        return '토요일'
    else:
        return '평일'
    
df_Nop['요일구분'] = df_Nop['요일구분'].apply(change_day)


print(df_Nop.head())


df_Nop['수송일자']

new_df_Nop = df_Nop.groupby(['역명','호선','승하차구분','요일구분']).mean(numeric_only=True)
new_df_Nop.drop(['연번'],axis=1,inplace=True)


new_df_Nop.to_csv("test_dataset/new_df_Nop.csv",encoding="cp949")