import pandas as pd
import numpy as np

# CSV 파일 읽기
dataset_Congestion = pd.read_csv('test_dataset/congestion_nan.csv', encoding='cp949')

time_column = ["5시30분","6시00분","6시30분","7시00분","7시30분","8시00분","8시30분","9시00분","9시30분","10시00분","10시30분","11시00분","11시30분","12시00분","12시30분","13시00분","13시30분","14시00분","14시30분","15시00분","15시30분","16시00분","16시30분","17시00분","17시30분","18시00분","18시30분","19시00분","19시30분","20시00분","20시30분","21시00분","21시30분","22시00분","22시30분","23시00분","23시30분","00시00분","00시30분"]


dataset_Congestion[time_column] = dataset_Congestion[time_column].replace(np.nan,0)


# 변환 비율 정의
conversion_factors = {1: 15.9, 2: 15.9, 3: 15.9, 4: 15.9, 5: 12.7, 6: 12.7, 7: 12.7, 8: 9.5}


# 각 퍼센트 값을 실수로 변환
for idx, row in dataset_Congestion.iterrows():
    line = row['호선']
    factor = conversion_factors.get(line, 1)  # 기본적으로 1을 곱하도록 설정
    for column in time_column:
        dataset_Congestion.at[idx, column] = int(row[column] * factor) # 소수점 안남기고 정수로 처리

# 결과 출력
print(dataset_Congestion)

# 데이터셋 저장 (필요시)
dataset_Congestion.to_csv('test_dataset/converted_dataset_Congestion2.csv', index=False, encoding='cp949')