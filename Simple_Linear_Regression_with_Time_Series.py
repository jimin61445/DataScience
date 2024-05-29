import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 데이터 로드
file_path = 'test_dataset/transformed_dataset_Congestion.csv'
data = pd.read_csv(file_path,encoding='cp949')

# 데이터 확인
print(data.head())

# 시간대 컬럼을 리스트로 만듭니다
time_columns = ['06시이전', '06-07시간대', '07-08시간대', '08-09시간대', '09-10시간대', '10-11시간대', '11-12시간대', '12-13시간대', '13-14시간대', '14-15시간대', '15-16시간대', '16-17시간대', '17-18시간대', '18-19시간대', '19-20시간대', '20-21시간대', '21-22시간대', '22-23시간대', '23-24시간대', '24시이후']

# 각 호선과 상하행 구분에 따라 시계열 데이터를 나눕니다.
lines = data['호선'].unique()
directions = data['상하구분'].unique()

# 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 모든 호선과 방향에 대해 시계열 데이터 시각화
for line in lines:
    for direction in directions:
        subset = data[(data['호선'] == line) & (data['상하구분'] == direction)]
        if not subset.empty:
            plt.figure(figsize=(15, 6))
            for index, row in subset.iterrows():
                plt.plot(time_columns, row[time_columns], marker='o', label=f'{row["출발역"]} {direction} {row["연번"]}')
            plt.title(f'{line}호선 {direction} 시간대별 탑승객 수')
            plt.xlabel('시간대')
            plt.ylabel('탑승객 수')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.show()

# 예시로 1호선 상선의 데이터를 사용하여 시계열 모델링
for line in lines:
    for direction in directions:
        subset = data[(data['호선'] == line) & (data['상하구분'] == direction)]
        if not subset.empty:
            first_row = subset.iloc[0]
            time_series = first_row[time_columns].values

            # ARIMA 모델 적용
            model = ARIMA(time_series, order=(1, 1, 1))
            model_fit = model.fit()

            # 예측
            forecast = model_fit.forecast(steps=5)  # 5시간대 예측
            print(f'{line}호선 {direction} 예측 결과: {forecast}')

            # 결과 시각화
            plt.figure(figsize=(10, 6))
            plt.plot(time_columns, time_series, marker='o', label='실제 데이터')
            plt.plot(range(len(time_columns), len(time_columns) + len(forecast)), forecast, marker='o', color='red', label='예측 데이터')
            plt.title(f'{line}호선 {direction} 시간대별 탑승객 수 예측')
            plt.xlabel('시간대')
            plt.ylabel('탑승객 수')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.show()
