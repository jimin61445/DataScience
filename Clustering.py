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



# 시간대 컬럼을 리스트로 만듭니다
time_columns = ['06시이전', '06-07시간대', '07-08시간대', '08-09시간대', '09-10시간대', '10-11시간대', '11-12시간대', '12-13시간대', '13-14시간대', '14-15시간대', '15-16시간대', '16-17시간대', '17-18시간대', '18-19시간대', '19-20시간대', '20-21시간대', '21-22시간대', '22-23시간대', '23-24시간대', '24시이후']

df_name = pd.read_csv('test_dataset/station_name.csv',encoding='cp949')

df = pd.read_csv('test_dataset/standard_row.csv',encoding='cp949')

norm =  Normalizer()

# df = df.drop(df.columns[:3], axis=1)
new_df = df.groupby(['역번호','승하차구분']).mean(numeric_only=True)
new_df = new_df.groupby('역번호').apply(lambda x: x[time_columns].iloc[0] - x[time_columns].iloc[1])
new_df[time_columns] = norm.fit_transform(new_df[time_columns])


merged_df = pd.merge(new_df,df_name,on='역번호',how='left')


print(merged_df)

# 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


for idx,val in new_df.iterrows():
    plt.plot(val.index, val.values,"k-", alpha=0.2)

plt.xlabel('시간대')
plt.ylabel('값')
plt.title(idx)
plt.legend()
plt.show()




# selected_rows = merged_df[merged_df['역번호']<500]

X = merged_df[time_columns].values



# # 클러스터 수 범위
# min_clusters = 2
# max_clusters = 10

# # 각 클러스터 수에 대한 실루엣 스코어 계산
# inertia = []
# silhouette_scores = []
# for k in range(min_clusters, max_clusters + 1):
#     kmeans = TimeSeriesKMeans(n_clusters=k, metric="euclidean", verbose=False, random_state=0)
#     y_pred = kmeans.fit_predict(X)
#     silhouette_scores.append(silhouette_score(X, y_pred))
#     inertia.append(kmeans.inertia_)


# # 시각화
# plt.plot(range(min_clusters, max_clusters + 1), inertia, marker='o')
# plt.xlabel('클러스터 수 (K)')
# plt.ylabel('inertia')
# plt.title('클러스터 수에 따른 Inertia 변화')
# plt.xticks(range(min_clusters, max_clusters + 1))
# plt.grid(True)
# plt.show()


# # 최적의 K 값 찾기
# best_k = np.argmax(silhouette_scores) + min_clusters
# print(f"최적의 클러스터 수 (K): {best_k}")

# K-means 클러스터링
n_clusters = 2  # 클러스터 수
kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", verbose=False, random_state=0)
y_pred = kmeans.fit_predict(X)

# merged_df['cluster'] = y_pred

# 클러스터링 결과 출력
print("클러스터링 결과:")
for cluster_idx in range(n_clusters):
    print(f"클러스터 {cluster_idx + 1}에 속하는 데이터 개수:", np.sum(y_pred == cluster_idx))
    
# 클러스터링 결과 시각화
for cluster_idx in range(n_clusters):
    plt.figure(figsize=(8, 6))
    for series_idx in range(len(X[y_pred == cluster_idx])):
        plt.plot(time_columns, X[y_pred == cluster_idx][series_idx], "k-", alpha=0.2)
    plt.plot(time_columns, kmeans.cluster_centers_[cluster_idx], "r-", linewidth=2)
    plt.title(f"Cluster {cluster_idx + 1}")
    plt.xlabel('시간대')
    plt.ylabel('표준화된 탑승객 수')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def visualize_silhouette(cluster_lists,X_features):
    n_cols = len(cluster_lists)
    
    fig, axs = plt.subplots(figsize=(4*n_cols, 10), nrows=2, ncols=n_cols)
    
    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 실루엣 개수 시각화
    for ind, n_cluster in enumerate(cluster_lists):
        cluster = TimeSeriesKMeans(n_clusters=n_cluster, metric="euclidean", verbose=False, random_state=0)
        y_pred = cluster.fit_predict(X)
        centers = cluster.cluster_centers_

        sil_avg = silhouette_score(X_features,y_pred)
        sil_values = silhouette_samples(X_features,y_pred)

        y_lower = 10
        axs[0,ind].set_title('Number of Cluster : '+ str(n_cluster)+'\n' \
                          'Silhouette Score :' + str(round(sil_avg,3)) )
        axs[0,ind].set_xlabel("The silhouette coefficient values")
        axs[0,ind].set_ylabel("Cluster label")
        axs[0,ind].set_xlim([-0.1, 1])
        axs[0,ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
        axs[0,ind].set_yticks([])  # Clear the yaxis labels / ticks
        axs[0,ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        
        # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현. 
        for i in range(n_cluster):
            ith_cluster_sil_values = sil_values[y_pred==i]
            ith_cluster_sil_values.sort()
            
            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = cm.nipy_spectral(float(i) / n_cluster)
            axs[0,ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \
                                facecolor=color, edgecolor=color, alpha=0.7)
            axs[0,ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

            # 클러스터링된 데이터 시각화
            axs[1,ind].scatter(X_features[:, 0], X_features[:, 1], marker='.', s=30, lw=0, alpha=0.7, \
                c=y_pred)
            axs[1,ind].set_title("Clustered data")
            axs[1,ind].set_xlabel("Feature space for the 1st feature")
            axs[1,ind].set_ylabel("Feature space for the 2nd feature")  

        # 군집별 중심 위치 좌표 시각화 
        unique_labels = np.unique(y_pred)
        for label in unique_labels:
            center_x_y = centers[label]
            axs[1,ind].scatter(x=center_x_y[0], y=center_x_y[1], s=70, color='k', edgecolor='k', 
                        marker='$%d$' % label)
            
        axs[0,ind].axvline(x=sil_avg, color="red", linestyle="--")

# visualize_silhouette([2,3,4,5],X_scaled)
# plt.show()
# print(merged_df)

merged_df.to_csv('test_dataset/clustered_data.csv', encoding='cp949')