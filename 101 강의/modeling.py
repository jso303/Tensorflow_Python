import pandas as pd
import tensorflow as tf

# 데이터 불러오기
파일경로 = "./csv/lemonade.csv"
데이터 = pd.read_csv(파일경로)

# 독립변수와 종속변수의 분리
독립 = 데이터[['온도']]
종속 = 데이터[['판매량']]

X = tf.keras.layers.Input(shape=[1])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X,Y)
model.compile(loss='mse')

# fit(원인, 결과, 반복학습횟수)
# loss : 학습의 정확도 수치 (0에 가까울 수록 정확도가 높음)
model.fit(독립, 종속, epochs=1000)

print("Predictions:", model.predict([[30]]))