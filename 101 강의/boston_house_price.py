# 보스턴 집값 예측하기 모델
# 변수값이 너무 많아 loss가 24 이하로 줄어들지 않는다

import tensorflow as tf
import pandas as pd

file = "./csv/boston.csv"
data = pd.read_csv(file)

# 집값에 영향을 끼치는 변수들
input = data[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
            'ptratio', 'b', 'lstat']]

# 집값의 중앙값
output = data[['medv']]

X = tf.keras.layers.Input(shape=[13])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X,Y)
model.compile(loss='mse')

# verbose=0 은 학습 출력을 끄고 훈련을 시킴.
model.fit(input, output, epochs=1000, verbose=0)
# 학습 loss를 보기 위해 verbose 없이 훈련 2회
model.fit(input, output, epochs=2)

# 5개 집에 대해서 예측 하기
print("Predictions:", model.predict(input[0:5]))

# 실제 정답 확인하기
print(output[0:5])