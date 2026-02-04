import pandas as pd

csv_path = r"C:\Users\chocy\AI_07_basic\runs\detect\pill_resume_stable\results.csv"
df = pd.read_csv(csv_path)

print(df.head())      # 앞부분 보기
print(df.tail())      # 마지막 부분 보기
print(df.columns)     # 컬럼 목록
df[['epoch', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(B)', 'metrics/recall(B)']]
