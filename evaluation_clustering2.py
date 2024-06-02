import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer

from sklearn.metrics import silhouette_score,silhouette_samples
import matplotlib.cm as cm

# 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False



# 시간대 컬럼을 리스트로 만듭니다
time_columns = ['06시이전', '06-07시간대', '07-08시간대', '08-09시간대', '09-10시간대', '10-11시간대', '11-12시간대', '12-13시간대', '13-14시간대', '14-15시간대', '15-16시간대', '16-17시간대', '17-18시간대', '18-19시간대', '19-20시간대', '20-21시간대', '21-22시간대', '22-23시간대', '23-24시간대', '24시이후']

df = pd.read_csv('test_dataset/before_cluster2.csv',encoding='cp949')
df_name = pd.read_csv('test_dataset/station_name.csv',encoding='cp949')

df = df.drop(df.columns[:1], axis=1)

print(df)

from sklearn.cluster import KMeans


# 클러스터 수 범위
min_clusters = 2
max_clusters = 10

X = df[['승차', '하차', '혼잡도']].values

# 각 클러스터 수에 대한 Inertia, 실루엣 스코어 계산
inertia = []
silhouette_scores = []
for k in range(min_clusters, max_clusters + 1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_pred = kmeans.fit_predict(df[['승차', '하차', '혼잡도']])
    silhouette_scores.append(silhouette_score(X, y_pred))
    inertia.append(kmeans.inertia_)


# 시각화
plt.plot(range(min_clusters, max_clusters + 1), inertia, marker='o')
plt.xlabel('클러스터 수 (K)')
plt.ylabel('inertia')
plt.title('클러스터 수에 따른 Inertia 변화')
plt.xticks(range(min_clusters, max_clusters + 1))
plt.grid(True)
plt.show()

# 시각화
plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o')
plt.xlabel('클러스터 수 (K)')
plt.ylabel('inertia')
plt.title('클러스터 수에 따른 실루엣 스코어 변화')
plt.xticks(range(min_clusters, max_clusters + 1))
plt.grid(True)
plt.show()



# 최적의 K 값 찾기
best_k = np.argmax(silhouette_scores) + min_clusters
print(f"최적의 클러스터 수 (K): {best_k}")


kmeans = KMeans(n_clusters=best_k, random_state=42)

kmeans.fit(df[['승차', '하차', '혼잡도']])

df['클러스터'] = kmeans.labels_


from mpl_toolkits.mplot3d import Axes3D

# 시각화
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 각 클러스터에 해당하는 데이터를 따로 표시
for cluster in df['클러스터'].unique():
    cluster_data = df[df['클러스터'] == cluster]
    ax.scatter(cluster_data['승차'], cluster_data['하차'], cluster_data['혼잡도'], label=f'클러스터 {cluster}')

ax.set_xlabel('승차')
ax.set_ylabel('하차')
ax.set_zlabel('혼잡도')
ax.set_title('K-평균 클러스터링 결과')
ax.legend()
plt.show()

merged_df = pd.merge(df_name,df,on='역번호')
merged_df = merged_df.drop(columns=["Unnamed: 0"])
merged_df.to_csv("test_dataset/cluster2.csv",encoding='cp949')

print(merged_df)

cluster_groups = merged_df.groupby('클러스터')




cluster_station_lists = {}

for cluster, group in cluster_groups:
    station_names = group['역명'].tolist()
    cluster_station_lists[cluster] = station_names

print(cluster_station_lists)