# 표를 다루는 도구 pandas

import pandas as pd

# 데이터 불러오기
파일경로 = "./csv/lemonade.csv"
데이터 = pd.read_csv(파일경로)

# 독립변수와 종속변수의 분리
독립 = 데이터[['온도']]
종속 = 데이터[['판매량']]

# 데이터 모양 확인
print(독립.shape, 종속.shape)
print(독립, 종속)