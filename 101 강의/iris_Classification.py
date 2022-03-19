# 아이리스 품종 분류하기
# 범주형 데이터 다루기

# 머신러닝 회귀와 분류를 나누는 기준은 종속변수(output)의 데이터 타입이다.
# 종속변수 : 양적데이터 = 회귀문제
# 종속변수 : 범주형데이터 = 분류문제

import tensorflow as tf
import pandas as pd

file = './csv/iris.csv'
data = pd.read_csv(file)

# 데이터 내의 범주형 변수만 골라서 원핫인코딩 시켜줌
# 원핫인코딩은 범주형 데이터를 수식에 사용할 수 있도록 범주를 칼럼으로 만들어
# 0과 1의 데이터로 분류해줌
iris = pd.get_dummies(data)
print(iris)

input = iris[['꽃잎길이','꽃잎폭','꽃받침길이','꽃받침폭']]
output = iris[['품종_setosa', '품종_versicolor', '품종_virginica']]

print(input.shape, output.shape)

# 모델 제작 (x,y 갯수는 위의 shape로 확인한 4와 3 사용)
X = tf.keras.layers.Input(shape=[4])
# softmax 는 0~100%의 확률로 예측함
Y = tf.keras.layers.Dense(3, activation='softmax')(X)
model = tf.keras.models.Model(X, Y)

# 분류에 사용하는 loss는 crossentropy
# 회귀에 사용하는 loss는 mse
model.compile(loss='categorical_crossentropy')

# 모델 학습시키기
model.fit(input, output, epochs=100)

print(model.predict(input[:5]))
print(model.predict(output[:5]))
