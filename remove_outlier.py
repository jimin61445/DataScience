import pandas as pd
import numpy as np

# 데이터셋 불러오기
df = pd.read_csv("test_dataset/after_handling_nan.csv", encoding='cp949')
# 필요없는 앞의 두 열 제거
df = df.drop(df.columns[:3], axis=1)

# 시간대 데이터 배열로 저장
timeColumns = np.array(['06시이전', '06-07시간대', '07-08시간대', '08-09시간대', '09-10시간대', '10-11시간대', '12-13시간대', '13-14시간대', '14-15시간대', '15-16시간대', '16-17시간대', '17-18시간대', '18-19시간대', '19-20시간대', '20-21시간대', '21-22시간대', '22-23시간대', '23-24시간대', '24시이후'])

# '역번호'와 '승하차구분'을 기준으로 그룹화
grouped_data = df.groupby(['역번호', '승하차구분'])
final_df = pd.DataFrame()

# 최종 몇 개 지워졌는지 알기 위해 변수 생성
total_removed_num = 0

# 그룹 데이터 순회
# group_key가 (150, 0)이면 역번호 150, 승차 / (150, 1)이면 역번호 150, 하차 데이터
for group_key, group_df in grouped_data:
    print(f"[{group_key}번째 그룹 값]")
    # 시간대별로 이상치 구하고 제거
    for t in timeColumns:
        print(f"{t} 시간의 이상치 제거")
        # 시간대의 평균, 표준편차
        mean = np.mean(group_df[t])
        std = np.std(group_df[t])
        # 신뢰구간 95%로 Z-Score 구해서 이상치의 인덱스를 구함
        idx = group_df[(abs((group_df[t]-mean)/std))>1.96].index
        # 값 확인을 위한 출력문
        print(f"[max] {np.max(group_df[t])}")
        print(f"[min] {np.min(group_df[t])}")
        print(f"[mean] {mean}")
        print(f"[std] {std}")
        print(f"[outlier index] {idx}")
        print(f"[removed num] {len(idx)}")
        # 이상치 개수 누적
        total_removed_num += len(idx)
        # 그룹에서 이상치들을 drop
        group_df = group_df.drop(idx)
        print()
    # 존재하는 시간대 다 돌았으면 파이널에 그 그룹 데이터 누적
    final_df = pd.concat([final_df,group_df])
# 누적된 최종 df 값을 csv 파일로 저장
final_df.to_csv(f"test_dataset/remove_outlier_final.csv",encoding='cp949')
# 최종 몇 개의 이상치가 삭제되었는지 출력
print(f"[total removed num] {total_removed_num}개 삭제됨")