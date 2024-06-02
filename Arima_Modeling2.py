import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# 데이터 로드
file_path = 'test_dataset/converted_dataset_Congestion.csv'
data = pd.read_csv(file_path, encoding='cp949')
print(data.info())

# 데이터 확인
print(data.head())

# 시간대 컬럼을 리스트로 만듭니다
time_columns = ['5시30분', '6시00분', '6시30분', '7시00분', '7시30분', '8시00분', '8시30분', '9시00분', 
                '9시30분', '10시00분', '10시30분', '11시00분', '11시30분', '12시00분', '12시30분', '13시00분', 
                '13시30분', '14시00분', '14시30분', '15시00분', '15시30분', '16시00분', '16시30분', '17시00분', 
                '17시30분', '18시00분', '18시30분', '19시00분', '19시30분', '20시00분', '20시30분', '21시00분', 
                '21시30분', '22시00분', '22시30분', '23시00분', '23시30분', '00시00분', '00시30분']

# 필요한 컬럼만 선택
data = data[['호선', '상하구분'] + time_columns]

# 결측치 확인
print(data.isnull().sum())

# 결측치가 있을 경우 이를 처리 (예: 0으로 채우기)
data = data.fillna(0)

# 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 각 호선과 상하행 구분에 따라 시계열 데이터를 나눕니다.
lines = data['호선'].unique()
directions = data['상하구분'].unique()

# 승하차구분을 '상차'와 '하차'로 변경하는 함수 정의
def map_boarding_alighting(value):
    if value == '상선':
        return '상차'
    else:
        return '하차'

# ARIMA 모델 적용 및 예측
for line in lines:
    for direction in directions:
        subset = data[(data['호선'] == line) & (data['상하구분'] == direction)]
        if not subset.empty:
            first_row = subset.iloc[0]
            time_series = first_row[time_columns].values
            time_series = np.array(time_series, dtype=float)

            # ARIMA 모델 적용
            model = ARIMA(time_series, order=(5, 1, 0))
            model_fit = model.fit()

            # 예측
            forecast = model_fit.forecast(steps=len(time_columns))
            print(f'{line}호선 {direction} 예측 결과: {forecast}')

            # 결과 시각화
            plt.figure(figsize=(10, 6))
            plt.plot(time_columns, time_series, marker='o', label='실제 데이터')
            plt.plot(time_columns, forecast, marker='o', color='red', label='예측 데이터')
            plt.title(f'{line}호선 {map_boarding_alighting(direction)} 시간대별 탑승객 수 예측')
            plt.xlabel('시간대')
            plt.ylabel('탑승객 수')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.show()