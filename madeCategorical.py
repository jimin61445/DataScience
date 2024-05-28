import pandas as pd 
import numpy as np

df_Nop = pd.read_csv("original_dataset/dataset_Nop.csv",encoding='cp949')

df_Nop.drop(['역명'],axis=1,inplace=True)

df_Nop['호선'] = df_Nop['호선'].str.replace('호선','').astype(int)

df_Nop['호선'] = df_Nop['호선'].astype('category')

df_Nop['승하차구분'] = df_Nop['승하차구분'].astype('category').cat.codes
# df_Nop['요일구분'] = df_Nop['요일구분'].astype('category').cat.codes

df_Nop.to_csv('test_dataset/categorical_dataset.csv',encoding='cp949')

