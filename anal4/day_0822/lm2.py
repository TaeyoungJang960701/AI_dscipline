# 카페 게시글 파이선/애널리시스 29번 게시글에 선형회귀 글로 수업 시작

# 선형회귀 모델식 계산 - 최소제곱법(ols)으로 
# w = wx + b 형태의 추세식 파라미터 w와 b를 추정

import numpy as np

class MySimpleLinearRegression:
    def __init__(self):
        self.w = None
        self.b = None   
    
    def fit(self, x:np.ndarray, y:np.ndarray):
        # ols로 w와 b를 추정해보자
        # x값은 독립변수로 여기선 키가 되겠다
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x-x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        self.w = numerator / denominator
        # 위 세 줄이 바로 최소제곱법식 그대로 나타낸거
        self.b = y_mean - (self.w * x_mean)     # 절편 b 구하는거
        
    def predict(self, x:np.ndarray):
        return self.w * x + self.b




def main():
    np.random.seed(42)  # 쌤이랑 결과값 통일을 위해 seed값 사용
    # 임의의 성인 남성 10명의 키, 몸무게 자료를 사용
    # 가우시안 분포로 가자
    x_heights = np.random.normal(175, 5, 10)    # 평균 175, 분산 5, 데이터 10개
    y_weights = np.random.normal(70, 10, 10)

    # 최소제곱법을 수행하는 클래스 객체 생성
    model = MySimpleLinearRegression()
    # 학습 수행 - fit()
    model.fit(x_heights, y_weights)

    # 추정된 w와 b 출력
    print('w : ', model.w)
    print('b : ', model.b) # y = wx + b

    # 예측값 확인
    y_pred = model.predict(x_heights)
    # print(y_pred)

    print('실제 몸무게와 예측 몸무게 비교')
    for i in range(len(x_heights)):
        print(f'키{x_heights[i]:.2f}cm, 실제 몸무게:{y_weights[i]:.2f}kg')
    print('미지의 남성 키 199의 몸무게는 ?? ', model.predict(199))

if __name__ == '__main__':
    main()

