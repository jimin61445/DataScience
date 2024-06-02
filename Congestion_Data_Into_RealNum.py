import pandas as pd

# CSV 파일 읽기
dataset_Congestion = pd.read_csv('original_dataset/dataset_Congestion.csv', encoding='cp949')

print(dataset_Congestion)

# 변환 비율 정의
conversion_factors = {1: 15.9, 2: 15.9, 3: 15.9, 4: 15.9, 5: 12.7, 6: 12.7, 7: 12.7, 8: 9.5}

# 시간 데이터 열만 선택
time_columns = dataset_Congestion.columns[6:]

# 각 퍼센트 값을 실수로 변환
for idx, row in dataset_Congestion.iterrows():
    line = row['호선']
    factor = conversion_factors.get(line, 1)  # 기본적으로 1을 곱하도록 설정
    for column in time_columns:
        dataset_Congestion.at[idx, column] = int(row[column] * factor) # 소수점 안남기고 정수로 처리

# 결과 출력
print(dataset_Congestion)

# 데이터셋 저장 (필요시)
dataset_Congestion.to_csv('test_dataset/converted_dataset_Congestion.csv', index=False, encoding='cp949')