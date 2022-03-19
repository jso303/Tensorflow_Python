# 여러번 학습시 과적합으로 오히려 loss가 증가하니 주의
# 과적합 해결을 위해서는 데이터 양을 늘리거나 가중치에 제한을 주어야함

import tensorflow as tf
import pandas as pd

file = "./csv/lemonade.csv"
data = pd.read_csv(file)

# 데이터 확인
# print(data)

input = data[['온도']]
output = data[['판매량']]

X = tf.keras.layers.Input(shape=[1])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X,Y)
model.compile(loss='mse')

# verbose=0 은 학습 출력을 끄고 훈련을 시킴.
model.fit(input, output, epochs=1000, verbose=0)
# 학습 loss를 보기 위해 verbose 없이 훈련 2회
model.fit(input, output, epochs=2)

# 입력값에 대해 에측 출력값 확인해보기
print("Predictions:", model.predict(input))

# 임의의 입력값에 대해 예측 출력값 확인하기
print("Predictions:", model.predict([[30]]))