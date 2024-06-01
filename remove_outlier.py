import pandas as pd
import numpy as np

# Read dataset
df = pd.read_csv("test_dataset/after_handling_nan.csv", encoding='cp949')
# Remove unused columns
df = df.drop(df.columns[:3], axis=1)

# Store time slot data to np.array
time_columns = np.array(['06시이전', '06-07시간대', '07-08시간대', '08-09시간대', '09-10시간대', '10-11시간대', '12-13시간대', '13-14시간대', '14-15시간대', '15-16시간대', '16-17시간대', '17-18시간대', '18-19시간대', '19-20시간대', '20-21시간대', '21-22시간대', '22-23시간대', '23-24시간대', '24시이후'])

# Grouping based on '역번호' and '승하차구분'
grouped_data = df.groupby(['역번호', '승하차구분'])
# Make empty df for final
final_df = pd.DataFrame()

# To know how many data removed finally
total_removed_num = 0

# Traversal group data
# If group_key is (150, 0), 역번호 value is 150 and 승차 / If (150, 1), 역번호 value is 150 and 하차 data
for group_key, group_df in grouped_data:
    print(f"[{group_key}번째 그룹 값]")
    # Remove outlier by time slot
    for t in time_columns:
        print(f"{t} 시간의 이상치 제거")
        # Avg and Std of time slot
        mean = np.mean(group_df[t])
        std = np.std(group_df[t])
        # Get index by calculate Z-Score with 95% confidence interval
        idx = group_df[(abs((group_df[t]-mean)/std))>1.96].index
        # For checking value
        print(f"[max] {np.max(group_df[t])}")
        print(f"[min] {np.min(group_df[t])}")
        print(f"[mean] {mean}")
        print(f"[std] {std}")
        print(f"[outlier index] {idx}")
        print(f"[removed num] {len(idx)}")
        # Accumulate outlier number
        total_removed_num += len(idx)
        # Drop outliers from group
        group_df = group_df.drop(idx)
        print()
    # Append group data to final df
    final_df = pd.concat([final_df,group_df])
# Save csv file
final_df.to_csv(f"test_dataset/remove_outlier_final.csv",encoding='cp949')
# Print how many outliers removed
print(f"[total removed num] {total_removed_num}개 삭제됨")