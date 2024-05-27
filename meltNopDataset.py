import pandas as pd 
import numpy as np 

new_df_Nop = pd.read_csv("test_dataset/new_df_Nop.csv",encoding='cp949')

df_Nop_Melt = new_df_Nop.melt(id_vars = ['역명','호선','승하차구분','요일구분','역번호'],value_vars=['06시이전','06-07시간대','07-08시간대','08-09시간대','09-10시간대','10-11시간대','11-12시간대','12-13시간대','13-14시간대','14-15시간대','15-16시간대','16-17시간대','17-18시간대','19-20시간대','20-21시간대','21-22시간대','22-23시간대','23-24시간대','24시이후'])


df_Nop_Melt.to_csv('test_dataset/melt_df_Nop.csv',encoding='cp949')

print(df_Nop_Melt)