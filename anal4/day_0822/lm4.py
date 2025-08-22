# 방법 4 : linregress 이것도 model 생성 O
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IQ에 따른 시험 점수 값 예측
score_iq = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/score_iq.csv')
print(score_iq.head(3))
print(score_iq.info())
print('-' * 100)

x = score_iq.iq
y = score_iq.score

# 상관 계수 확인
print(np.corrcoef(x,y)[0,1])    # 0.8822203446134701 
# 이건 양의 1에 가까운 수이기에 아주 쓸만한 상관 관계가 잇는것같애 쓸만한 데이터다
print('-' * 100)
print(score_iq.corr())
print('-' * 100)

# plt.scatter(x,y)
# plt.show()

model = stats.linregress(x,y)
print(model)
# LinregressResult(slope=np.float64(0.6514309527270075), 
# intercept=np.float64(-2.8564471221974657), 
# rvalue=np.float64(0.8822203446134699),            # 결정계수
# pvalue=np.float64(2.8476895206683644e-50),        
# 0.05보다 작은 피밸류 - 이 모델은 두 변수 간의 인과관계가 잇다, 그리고 이 데이터는 유의미하다
# stderr=np.float64(0.028577934409305443), 
# intercept_stderr=np.float64(3.546211918048538))
print('기울기 : ', model.slope)            # 0.6514309527270075
print('절편 : ', model.intercept)          # -2.8564471221974657
print('R² - 결정계수 : ', model.rvalue)    # 0.8822203446134699     -> 독립변수가 종속변수를 88%정도 설명하고 있다
print('p-value : ', model.pvalue)         # 2.8476895206683644e-50 < 0.05 이므로 현재 모델은 유의하다.(독립변수와 종속변수)
print('표준오차 : ', model.stderr)         # 0.028577934409305443
# y = wx + b  => 0.6514309527270075 * x + (-2.8564471221974657)

# plt.scatter(x,y)
# plt.plot(x,model.slope * x + model.intercept)
# plt.show()

# 점수 예측
print('점수 예측 : ', model.slope * 80 + model.intercept)
print('점수 예측 : ', model.slope * 120 + model.intercept)
# 이 linregress 모듈은 predict 클래스가 없나봐
print('점수 예측 : ', 
      np.polyval([model.slope, model.intercept], np.array(score_iq['iq'][:5])))

print()
newdf = pd.DataFrame({'iq':[55,66,77,88,150]})
print('점수 예측 : ', 
      np.polyval([model.slope, model.intercept], np.array(newdf['iq'][:5])))
# 지금 하는건 계속 흩어진 데이터를 보고 1차 방정식 그래프를 잘 그어서 인풋에 대한 아웃풋을 예측하려는거야
