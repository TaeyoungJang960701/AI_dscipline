# 선형회귀모델 추섹식 계산
import numpy as np

class LinearRegressionTest:
  def __init__(self, learning_rate, epochs):
    self.w = None
    self.b = None
    self.learning_rate = learning_rate
    self.epochs = epochs

  def fit(self, x:np.ndarray, y:np.ndarray):
    # 경사하강법(Gradient Descent Method)으로 w,b를 학습
    # parameter 초기화
    self.w = np.random.uniform(-2, 2)
    self.b = np.random.uniform(-2, 2)

    n = len(x)

    for epoch in range(self.epochs):
      y_pred = self.w * x + self.b        # 예측값
      loss = np.mean((y - y_pred) ** 2)   # 손실

      # 경사하강법을 하고 있는거야 이 두줄로
      dw = (-2 / n) * np.sum(x * (y - y_pred))    # 경사 계산, 편미분
      db = (-2 / n) * np.sum(y - y_pred)

      # 여기서 새로 갱신하고 있는거지
      self.w -= self.learning_rate * dw
      self.b -= self.learning_rate * db

      # 학습상태 출력
      if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1} / {self.epochs} - Loss: {loss:.5f}, w: {self.w:.5f}, b: {self.b:.5f}')

  def predict(self, x:np.ndarray):
    return self.w * x + self.b

def main():
  np.random.seed(42)

  # 이 내용들이 feature야
  x_heights = np.random.normal(175, 10, 30)
  true_w = 0.7
  true_b = -55
  noise = np.random.normal(0, 5, 30)

  # 이 y가 label이야
  y_weights = true_w * x_heights + true_b + noise

  print(x_heights)
  print(y_weights)

  # scaling(표준화)
  x_mean = np.mean(x_heights)
  x_std = np.std(x_heights)
  y_mean = np.mean(y_weights)
  y_std = np.std(y_weights)

  x_height_scaled = (x_heights - x_mean) / x_std
  y_weight_saled  = (y_weights - y_mean) / y_std

# 모델학습
  model = LinearRegressionTest(learning_rate = 0.001, epochs = 1000)
# 여기서 인자를 두개밖에 안받았는데도 오류가 안나고있어 이건 왜냐
# 위에서 리니어리그레션 테스트에 대한 정의에서 self 포함해서 세개를 선었했잖아
# 그래서 자동으로 인자를, model 자기 자신을 받고 있는거야 사실은

  model.fit(x_height_scaled, y_weight_saled)

  # 예측
  y_pred_scaled = model.predict(x_height_scaled)

  # 예측 결과 역변환
  y_pred = (y_pred_scaled * y_std) + y_mean
  print('y_pred : ', y_pred)

  # 모델 성능(MSE, R^2) 계산
  mse = np.mean((y_weights - y_pred) ** 2)
  ss_tot = np.sum((y_weights - np.mean(y_weights)) ** 2)
  ss_res = np.sum((y_weights - y_pred) ** 2)
  r2 = 1 - (ss_res / ss_tot)

  print('학습결과 ------')
  print(f'추정된 기울기, w : {model.w:.4f}')
  print(f'추정된 편향, b : {model.b:.4f}')

  for i in range(len(x_heights)):
    print(f'키 : {x_heights[i]:.2f}cm, 몸무게 실제값 : {y_weights[i]:.2f}kg, 몸무게 예측값 : {y_pred[i]:.2f}kg')

  print(f'MSE(평균 제곱 오차) : {mse:.5f}')
  print(f'R^2(결정계수) : {r2:.5f}')

if __name__ == '__main__':
  main()


