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

df = pd.read_csv('test_dataset/merge_df.csv',encoding='cp949')

# 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def find_best(df,row):
    X = df[row].values
    score = []
    clusters = [2,3,4,5,6,7,8,9,10]
    metrics = ['euclidean','dtw','softdtw']
    maxiter = [50,100,200,300]
    for maxit in maxiter:
        print(maxit)
        for metric in metrics:
            print(metric)
            inertia = []
            silhouette_scores = []
            for cluster in clusters:
                print(cluster)
                kmeans = TimeSeriesKMeans(n_clusters=cluster, metric=metric,max_iter=maxit,random_state=0)
                y_pred = kmeans.fit_predict(X)
                silhouette_scores.append(silhouette_score(X, y_pred))
                inertia.append(kmeans.inertia_)
            plt.plot(range(min(clusters), (max(clusters) + 1)), silhouette_scores, marker='o')
            plt.xlabel('클러스터 수 (K)')
            plt.ylabel('실루엣 스코어')
            plt.title('클러스터 수에 따른 실루엣 스코어 변화')
            plt.xticks(range(min(clusters), (max(clusters) + 1)))
            plt.grid(True)
            plt.show()
            plt.plot(range(min(clusters), (max(clusters) + 1)), inertia, marker='o')
            plt.xlabel('클러스터 수 (K)')
            plt.ylabel('inertia')
            plt.title('클러스터 수에 따른 Inertia 변화')
            plt.xticks(range(min(clusters), (max(clusters) + 1)))
            plt.grid(True)
            plt.show()
            # 최적의 K 값 찾기
            best_k = np.argmax(silhouette_scores) + min(clusters)
            score.append([maxit,metric,best_k,max(silhouette_scores)])
    return max(score,key=lambda x:x[3])

print(find_best(df,time_columns))
