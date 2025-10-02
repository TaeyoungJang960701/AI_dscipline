# 카트폴(CartPole : 수레차 막대 버티기)
# Q-learning : (off-policy Td Control) 방식으로 카트폴 환경을 학습하기
# MDP(Markov Decision Process) 기반의 강화학습 알고리즘
# MDP의 5가지 구성요소
# S(State)                      # 환경의 상태
# A(Action)                     # 에이전트가 선택할 수 있는 행동
# R(reward)                     # 상태-행동에 따른 보상
# P(transition Probability)     # 상태 전이 확률
# π(Policy)                    # 어떤 상태에서 어떤 행동을 할지를 결정

# !pip install gymnasium[classic-control]

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# 환경 설정
env = gym.make('CartPole-v1')   # 카트에 막대(pole)를 수직으로 세운 채 좌우로 움직여 균형을 유지하는 환경을 제공
print(env.observation_space)    # 

# 카트 위치, 카트 속도, 막대 기울기, 막대 각속도
obs_space_low = np.array([-2.4, -3.0, -0.5, -2.0])  
obs_space_high = np.array([2.4, 3.0, 0.5, 2.0])

# 상태 공간 이산화 수준 설정. Q-table은 연속적인 상태를 다룰 수 없음 구간으로 분할해야함(이거 안하면 큐테이블 터진대)
state_bins = [6, 12, 6, 12]
q_table = np.zeros(state_bins + [env.action_space.n])
# print(q_table, q_table.shape)       # (6, 12, 6, 12, 2)

# 상태 이산화 처리 함수
def discretize_state(state):
    ratios = (state - obs_space_low) / (obs_space_high - obs_space_low)
    # (0 - (-2.4)) / (2.4 - (-2.4)) = 0.5
    print('ratios : ', ratios)
    discrete = (ratios * state_bins).astype(int)    # 구간이 선택됨
    print('discrete : ', discrete)
    return tuple(np.clip(discrete, 0, np.array(state_bins) - 1))

# discretize_state 결과 실험
ex_state = np.array([1.0, 0.5, 0.1, -1.0])
dis_index = discretize_state(ex_state)
print('Q-table 인덱스 : ', dis_index)

# Q-learning의 하이퍼파라미터 설정
alpha = 0.1
gammer = 0.99
epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.05
episodes = 1000
reward_list = []
tragectories = []
