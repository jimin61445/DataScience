import pandas as pd

# 두 번째 데이터셋 동일하게 처리
dataset_Congestion = pd.read_csv('original_dataset/dataset_Congestion.csv', encoding='cp949')

# 필요한 열 추출 및 시간 단위로 합산
dataset_Congestion['06시이전'] = dataset_Congestion[['5시30분', '6시00분']].sum(axis=1)/2
dataset_Congestion['06-07시간대'] = dataset_Congestion[['6시30분', '7시00분']].sum(axis=1)/2
dataset_Congestion['07-08시간대'] = dataset_Congestion[['7시30분', '8시00분']].sum(axis=1)/2
dataset_Congestion['08-09시간대'] = dataset_Congestion[['8시30분', '9시00분']].sum(axis=1)/2
dataset_Congestion['09-10시간대'] = dataset_Congestion[['9시30분', '10시00분']].sum(axis=1)/2
dataset_Congestion['10-11시간대'] = dataset_Congestion[['10시30분', '11시00분']].sum(axis=1)/2
dataset_Congestion['11-12시간대'] = dataset_Congestion[['11시30분', '12시00분']].sum(axis=1)/2
dataset_Congestion['12-13시간대'] = dataset_Congestion[['12시30분', '13시00분']].sum(axis=1)/2
dataset_Congestion['13-14시간대'] = dataset_Congestion[['13시30분', '14시00분']].sum(axis=1)/2
dataset_Congestion['14-15시간대'] = dataset_Congestion[['14시30분', '15시00분']].sum(axis=1)/2
dataset_Congestion['15-16시간대'] = dataset_Congestion[['15시30분', '16시00분']].sum(axis=1)/2
dataset_Congestion['16-17시간대'] = dataset_Congestion[['16시30분', '17시00분']].sum(axis=1)/2
dataset_Congestion['17-18시간대'] = dataset_Congestion[['17시30분', '18시00분']].sum(axis=1)/2
dataset_Congestion['18-19시간대'] = dataset_Congestion[['18시30분', '19시00분']].sum(axis=1)/2
dataset_Congestion['19-20시간대'] = dataset_Congestion[['19시30분', '20시00분']].sum(axis=1)/2
dataset_Congestion['20-21시간대'] = dataset_Congestion[['20시30분', '21시00분']].sum(axis=1)/2
dataset_Congestion['21-22시간대'] = dataset_Congestion[['21시30분', '22시00분']].sum(axis=1)/2
dataset_Congestion['22-23시간대'] = dataset_Congestion[['22시30분', '23시00분']].sum(axis=1)/2
dataset_Congestion['23-24시간대'] = dataset_Congestion[['23시30분', '00시00분']].sum(axis=1)/2
dataset_Congestion['24시이후'] = dataset_Congestion[['00시30분']]

# 불필요한 30분 단위 열 삭제
dataset_Congestion = dataset_Congestion.drop(columns=[
    '5시30분', '6시00분', '6시30분', '7시00분', '7시30분', '8시00분', '8시30분', '9시00분',
    '9시30분', '10시00분', '10시30분', '11시00분', '11시30분', '12시00분', '12시30분', '13시00분',
    '13시30분', '14시00분', '14시30분', '15시00분', '15시30분', '16시00분', '16시30분', '17시00분',
    '17시30분', '18시00분', '18시30분', '19시00분', '19시30분', '20시00분', '20시30분', '21시00분',
    '21시30분', '22시00분', '22시30분', '23시00분', '23시30분', '00시00분', '00시30분'
])

# 결과 저장
dataset_Congestion.to_csv('test_dataset/transformed_dataset_Congestion.csv', index=False, encoding='cp949')