# 다항회귀분석 문제) 
# 데이터 로드 (Servo, UCI) : "https://archive.ics.uci.edu/ml/machine-learning-databases/servo/servo.data"
# cols = ["motor", "screw", "pgain", "vgain", "class"]

#  - 타깃/피처 (숫자만 사용: pgain, vgain)
#    x = df[["pgain", "vgain"]].astype(float)   
#    y = df["class"].values



#  - 학습/테스트 분할 ( 8:2 )
#  - 스케일링 (StandardScaler)
#  - 다항 특성 (degree=2) + LinearRegression 또는 Ridge 학습
#  - 성능 평가 
#  - 시각화

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder

plt.rc('font', family='Malgun Gothic')

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/servo/servo.data')
df.columns = ['motor','screw','pgain','vgain','class']

x = df[["pgain", "vgain"]].astype(float)
y = df["class"].values

# 문자열로 된 y를 숫자로 변환
le = LabelEncoder()
y_num = le.fit_transform(y)

# 선형 회귀 학습
model_lin = LinearRegression()
model_lin.fit(x, y_num)
y_lin_pred = model_lin.predict(x)
r2_lin = r2_score(y_num, y_lin_pred)

# 다항회귀 (degree=2)
poly = PolynomialFeatures(degree=2, include_bias=False)
x_quad = poly.fit_transform(x)
model_quad = LinearRegression()
model_quad.fit(x_quad, y_num)
y_quad_pred = model_quad.predict(x_quad)
r2_quad = r2_score(y_num, y_quad_pred)

# 시각화
x_fit = np.linspace(x.min(), x.max(), 100)[:, np.newaxis]  # 예측용 X
y_lin_fit = model_lin.predict(x_fit)
y_quad_fit = model_quad.predict(poly.transform(x_fit))

plt.scatter(x, y_num, color='lightgray', label='학습데이터')
plt.plot(x_fit, y_lin_fit, linestyle=':', color='b', lw=2, label=f'선형 회귀, R²={r2_lin:.2f}')
plt.plot(x_fit, y_quad_fit, linestyle='-', color='r', lw=2, label=f'다항(2차) 회귀, R²={r2_quad:.2f}')

plt.xlabel('pgain')
plt.ylabel('class(숫자)')
plt.legend()
plt.show()