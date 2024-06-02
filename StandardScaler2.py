import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def scale_csv(csv_file):
    # CSV 파일 읽기
    data = pd.read_csv(csv_file , encoding="cp949")
    # 데이터 프레임에서 숫자형 열 선택
    numeric_cols = data.iloc[:, 7:].columns
    # StandardScaler 객체 생성
    scaler = StandardScaler()
    # 선택된 열을 표준화
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data

# 스케일링된 데이터프레임 생성
scaled_data = scale_csv("test_dataset/transformed_dataset_Congestion.csv")

# 스케일된 데이터를 새로운 CSV 파일로 저장
scaled_data.to_csv("test_dataset/scaled_data_standard2.csv", index=False, encoding='cp949')

# 상관 행렬 계산
correlation_matrix = scaled_data.iloc[:, 7:].corr()

# 히트맵 그리기
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('상관 관계 히트맵')
plt.show()