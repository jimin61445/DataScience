import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def scale_csv_min_max(csv_file):
    # CSV 파일 읽기
    data = pd.read_csv(csv_file, encoding="cp949")
    # 6시 이전부터의 열만 선택
    numeric_cols = data.iloc[:, 7:].columns
    # MinMaxScaler 객체 생성
    scaler = MinMaxScaler()
    # 선택된 열을 min-max 스케일링
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data

# min-max 스케일링된 데이터프레임 생성
scaled_data_min_max = scale_csv_min_max("test_dataset/after_handling_nan.csv.csv")

# 스케일된 데이터를 새로운 CSV 파일로 저장
scaled_data_min_max.to_csv("test_dataset/scaled_data_min_max.csv", index=False, encoding='cp949')

# 상관 행렬 계산
correlation_matrix_min_max = scaled_data_min_max.iloc[:, 7:].corr()

# 히트맵 그리기
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_min_max, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('상관 관계 히트맵')
plt.show()