# Servo 데이터: 다항회귀(Ridge) + 시각화 (pgain 곡선)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures 
from sklearn.linear_model import Ridge,LinearRegression                  
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 로드 (UCI Servo)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/servo/servo.data" 
cols = ["motor", "screw", "pgain", "vgain", "class"]          # 컬럼명 지정 (원본에 헤더 없음)
# 모터 위치, 스크류 위치, P게인(비례 이득), V게인(속도 이득), 응답 시간(타깃)
df = pd.read_csv(url, names=cols)

# 특징/타깃 선택 (숫자만 사용: pgain, vgain)
x = df[["pgain", "vgain"]].astype(float)  # 입력 X: 두 수치 컬럼
y = df["class"].values.astype(float)      # 타깃 y: class 컬럼

# 학습/테스트 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42 
)

# 표준화(스케일러는 훈련셋으로만 fit)
scaler = StandardScaler()         # 평균0, 표준편차1로 변환하는 객체
x_train_s = scaler.fit_transform(x_train)  # 훈련셋으로 평균/표준편차 학습 + 변환
x_test_s  = scaler.transform(x_test)       # 테스트셋은 학습된 파라미터로만 변환

# 다항 특성 생성 + Ridge 회귀 학습
poly = PolynomialFeatures(degree=2, include_bias=False)  # 2차 항까지 생성(상수항 제외)
xtr_poly = poly.fit_transform(x_train_s)                 # 훈련셋: 다항 변환(2차, 상호작용 포함)
xte_poly = poly.transform(x_test_s)                      # 테스트셋: 같은 규칙으로 변환

# model = LinearRegression()
# PolynomialFeatures(degree=2)로 특성을 늘리면, 과적합 위험 !
model = Ridge(alpha=1.0)     # 그래서 릿지 회귀 모델(정규화 강도 alpha=1.0) 적당한 규제 기본값
model.fit(xtr_poly, y_train) 

# 성능 평가 (RMSE, R^2)
y_pred = model.predict(xte_poly)         # 테스트셋 예측
mse = mean_squared_error(y_test, y_pred) # MSE 계산
rmse = np.sqrt(mse)                      # RMSE(해석 쉬운 단위)
r2 = r2_score(y_test, y_pred)   

print("[TEST METRICS]")  
print(f"RMSE: {rmse:.3f}") 
print(f"R^2 : {r2:.4f}") 

# 1) 실제 vs 예측 산점도 → 모델 성능 시각적 확인
plt.figure(figsize=(6,6))                
plt.scatter(y_test, y_pred, alpha=0.6)  # 실제값-예측값 산점도
mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())  # 대각선 범위
plt.plot([mn, mx], [mn, mx], 'r--', lw=2, label="이상적 예측선")            # y=x 기준선(이상적)
plt.xlabel("실제 class")                     
plt.ylabel("예측 class")                     
plt.title("Servo (다항회귀, 테스트셋)") 
plt.legend()     
plt.grid(alpha=0.3)  
plt.show() 

# 2) pgain만 변화시키고 vgain은 평균 고정 → 다항회귀의 '곡선' 확인
x_line = np.linspace(x['pgain'].min(), x['pgain'].max(), 200)  # pgain 구간을 촘촘히 샘플링
vgain_mean = x['vgain'].mean()      # vgain은 평균값으로 고정
x_line_input = np.c_[x_line, np.full_like(x_line, vgain_mean)]  # (pgain, vgain=평균) 입력행렬

x_line_s = scaler.transform(x_line_input) # 스케일러로 표준화(훈련 파라미터 사용)
x_line_poly = poly.transform(x_line_s)    # 동일한 다항 변환 적용
y_line_pred = model.predict(x_line_poly)  # 곡선상 예측값 계산

plt.figure(figsize=(7,5))
plt.scatter(x['pgain'], y, alpha=0.45, label="실제 데이터")  # 원본 데이터 산점도(pgain vs class)
plt.plot(x_line, y_line_pred, 'r-', lw=2, \
         label="다항회귀 곡선(pgain, vgain=평균)")
plt.xlabel("pgain")
plt.ylabel("class")
plt.title("pgain에 따른 예측 곡선 (vgain=평균 고정)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()