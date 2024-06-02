import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import interp1d
from sklearn.preprocessing import Normalizer


time_columns = ['06시이전', '06-07시간대', '07-08시간대', '08-09시간대', '09-10시간대', '10-11시간대', '11-12시간대', '12-13시간대', '13-14시간대', '14-15시간대', '15-16시간대', '16-17시간대', '17-18시간대', '18-19시간대', '19-20시간대', '20-21시간대', '21-22시간대', '22-23시간대', '23-24시간대', '24시이후']
time_float = [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5]

normal =  Normalizer()


df_Nop = pd.read_csv("test_dataset/remove_outlier_final.csv",encoding='cp949')
df_Congestion = pd.read_csv("test_dataset/transformed_dataset_Congestion2.csv",encoding='cp949')

df_Congestion['상하구분'] = df_Congestion['상하구분'].replace("내선","상선")
df_Congestion['상하구분'] = df_Congestion['상하구분'].replace("외선","하선")


df_Congestion['상하구분'] = df_Congestion['상하구분'].astype('category').cat.codes


new_df = df_Nop.groupby(['역번호','승하차구분']).mean(numeric_only=True)
new_df_con = df_Congestion.groupby(['역번호']).mean(numeric_only=True)

# new_df[time_columns] = normal.fit_transform(new_df[time_columns])
# new_df_con[time_columns] = normal.fit_transform(new_df_con[time_columns])


print(new_df)
print(new_df_con)


# 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# fig, ax = plt.subplots(figsize=(10, 6))

# for idx in new_df.index.levels[0]:
#     sub_df = new_df.loc[idx]
#     values = sub_df.values.flatten()

#     # 정규분포 파라미터 추정
#     mu, std = norm.fit(values)
    
#     # 정규분포 곡선 생성
#     xmin, xmax = min(values), max(values)
#     x = np.linspace(xmin, xmax, 100)
#     p = norm.pdf(x, mu, std)
    
#     # 데이터 히스토그램 및 정규분포 곡선 그리기
#     ax.plot(x, p * max(values), label=f'Index {idx} - Mean: {mu:.2f}')

# ax.set_title('인덱스별 정규분포 추정 및 최고점')
# ax.set_xlabel('값')
# ax.set_ylabel('확률 밀도 함수')
# ax.legend()
# ax.grid(True)
# plt.show()

# 새로운 데이터프레임 생성
new_dfs = []
con_dfs = []
# 각 인덱스에서 최고점이 나오는 시간대를 새로운 데이터프레임에 추가
for idx, row in new_df.iterrows():
    values = row[time_columns].values
    mu, std = norm.fit(values)
    interpolation_function = interp1d(values, time_float, kind='linear')
    print(row[time_columns].values.max()-mu, idx,interpolation_function(row[time_columns].values.max()-mu))
    station = idx[0]  # 인덱스의 첫 번째 요소는 역번호
    direction = idx[1]  # 인덱스의 두 번째 요소는 승하차구분
    new_row = {'역번호': station, '승하차구분': direction, '정규분포 평균': interpolation_function(row[time_columns].values.max()-mu)}
    new_dfs.append(new_row)

for idx, row in new_df_con.iterrows():
    values = row[time_columns].values
    mu, std = norm.fit(values)
    interpolation_function = interp1d(values, time_float, kind='linear')
    best_time_index = np.argmax(values)
    best_time_float = time_float[best_time_index] -1  # 시간대를 실수형으로 변환
    station = idx  # 인덱스의 첫 번째 요소는 역번호
    new_row = {'역번호': station, '혼잡도': interpolation_function(row[time_columns].values.max()-mu)}
    con_dfs.append(new_row)
new_dfs = pd.DataFrame(new_dfs)
con_dfs = pd.DataFrame(con_dfs)


# 식별자별로 그룹화하여 값 결합
result = new_dfs.groupby('역번호').agg(lambda x: x.tolist()).reset_index()
result = result.drop(columns="승하차구분")

result['승차'] = result['정규분포 평균'].apply(lambda x: x[0])

result['하차'] = result['정규분포 평균'].apply(lambda x: x[1])

result.drop(columns=['정규분포 평균'], inplace=True)

merged_df = pd.merge(result, con_dfs, on='역번호')

print(merged_df)

merged_df.to_csv("test_dataset/before_cluster2.csv",encoding='cp949')

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)

kmeans.fit(merged_df[['승차', '하차', '혼잡도']])

merged_df['클러스터'] = kmeans.labels_

merged_df.to_csv("test_dataset/cluster2.csv",encoding='cp949')

from mpl_toolkits.mplot3d import Axes3D

# 시각화
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 각 클러스터에 해당하는 데이터를 따로 표시
for cluster in merged_df['클러스터'].unique():
    cluster_data = merged_df[merged_df['클러스터'] == cluster]
    ax.scatter(cluster_data['승차'], cluster_data['하차'], cluster_data['혼잡도'], label=f'클러스터 {cluster}')

ax.set_xlabel('승차')
ax.set_ylabel('하차')
ax.set_zlabel('혼잡도')
ax.set_title('K-평균 클러스터링 결과')
ax.legend()
plt.show()