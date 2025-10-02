# 기초적인 Q-learning 연습
# 완전한 환경 모델이 없다
# model-free 방식으로 벨만 방정식 기반(Q값 갱신)의 근사학습을 사용

# 1차원 선에서 좌/우 이동하며 보상받기

import numpy as np
import pandas as pd
import random

state_space = [0, 1, 2, 3, 4]
action_space = [-1, 1]

# Q-table 초기화    (상태 5 * 행동 2)
Q = np.zeros([len(state_space), len(action_space)])
print(Q)

# 하이퍼 파라미터
alpha = 0.5     # 학습률(learning-rate)
gamma = 0.9     # 할인률(discount-rate)
epsilon = 1.0  # 초기 탐험률(ε-greedy)
epsilon_min = 0.01
epsilon_decay = 0.95
episodes = 500

def get_reward(state):  # 보상 함수
    return 10 if state == 4 else 0

# 학습 루프 : 각 episode 마다 Q-table을 갱신하면서 목표상태(state==4)에 도달하기 위한 최적의 행동 정책 학습
for episode in range(episodes):
    state = 0

    for step in range(20):      # 하나의 에피소드 안에서 최대 20번 이동 시도
    # 행동 선택은 epsilon-greedy 정책을 따른다
        if random.random() < epsilon:
            action_index = random.randint(0, 1)     # 탐험(Exploration)
        else:
            action_index = np.argmax(Q[state])      # 이용(Exploitation)

        action = action_space[action_index]         # action_space[-1,1]
        next_state = state + action
        # print('next_state : ', next_state)

        # 유효 범위 밖은 이동 금지

        if next_state < 0 or next_state > 4:
            next_state = state

            reward = get_reward(next_state)

            # 벨만 방정식을 이용해서 Q-value를 갱신
            old_q = Q[state][action_index]      # 현재 추정된 Q-value
            next_max = np.max(Q[next_state])    # 다음 상태에서 가능한 모든 행동중 가장 큰 Q값을 선택
            Q[state][action_index] = old_q + alpha * (reward + gamma * next_max - old_q)

            state = next_state
            if reward == 10:
                break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# 결과 출력
print('학습딘 Q-table : ')
for s in range(5):
    print(f'State {s} : 왼쪽 = {Q[s][0]:.2f}, 오른쪽 = {Q[s][1]:.2f}')
