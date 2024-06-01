import pandas as pd
import matplotlib.pyplot as plt

# 데이터 로드
file_path = 'test_dataset/transformed_dataset_Congestion.csv'
data = pd.read_csv(file_path, encoding='cp949')

# 데이터 확인
print(data.head())
# 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
# 시간대 컬럼을 리스트로 만듭니다
time_columns = ['06시이전', '06-07시간대', '07-08시간대', '08-09시간대', '09-10시간대', '10-11시간대', '11-12시간대', '12-13시간대', '13-14시간대', '14-15시간대', '15-16시간대', '16-17시간대', '17-18시간대', '18-19시간대', '19-20시간대', '20-21시간대', '21-22시간대', '22-23시간대', '23-24시간대', '24시이후']

# 각 호선과 상하행 구분에 따라 시계열 데이터를 나눕니다.
lines = data['호선'].unique()
directions = data['상하구분'].unique()

# 탑승객 수를 한적함, 여유로움, 여유롭지만 자리가 몇개 없음, 자리가 꽉참, 서서감, 복잡으로 나누는 함수 정의
def classify_congestion(passenger_count):
    if passenger_count < 150:
        return 1 
    if 150 <= passenger_count < 300:
        return 2
    if 300 <= passenger_count < 500:
        return 3
    if 500 <= passenger_count < 540:
        return 4
    if 540 <= passenger_count < 900:
        return 5
    if 900 <= passenger_count:
        return 6

# 모든 호선과 방향에 대해 시간대별 탑승객 수 출력 및 시각화
for line in lines:
    for direction in directions:
        subset = data[(data['호선'] == line) & (data['상하구분'] == direction)]
        if not subset.empty:
            print(f'{line}호선 {direction} 탑승객 수 분류:')
            for index, row in subset.iterrows():
                classified_data = [classify_congestion(val) for val in row[time_columns]]
                print(f"{row['출발역']} {direction} {row['연번']}: {classified_data}")
            print('\n')



# 사용자가 선택한 출발역에 대한 탑승객 수 분류 및 시각화
def plot_congestion_for_station(station_name):
    subset = data[data['출발역'] == station_name]
    if not subset.empty:
        plt.figure(figsize=(15, 6))
        for direction in subset['상하구분'].unique():  # 상하구분 별로 그래프를 그립니다.
            selected_graph = None  # 선택한 그래프를 저장할 변수
            for index, row in subset[subset['상하구분'] == direction].iterrows():
                classified_data = [classify_congestion(val) for val in row[time_columns]]
                if selected_graph is None:
                    selected_graph, = plt.plot(time_columns, classified_data, marker='o', label=f'{row["호선"]}호선 {row["출발역"]} {direction} {row["연번"]}')
                else:
                    plt.plot(time_columns, classified_data, marker='o', alpha=0.01)  # 투명도 조절
            plt.title(f'{station_name} 탑승객 수 분류 ({direction})')
            plt.xlabel('시간대')
            plt.ylabel('탑승객 분류')
            plt.legend([selected_graph], [selected_graph.get_label()], loc='best')  # 선택한 그래프만 범례에 표시
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.show()

# 사용자 입력에 따른 서울역 탑승객 수 분류 및 시각화
station_name = '문정'  # 사용자가 선택한 역
plot_congestion_for_station(station_name)