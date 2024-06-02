import pandas as pd 
import numpy as np 

df = pd.read_csv("test_dataset/transformed_dataset_Congestion.csv",encoding="cp949")

nan_count = df.isnull().sum()

print(nan_count)
print(df)