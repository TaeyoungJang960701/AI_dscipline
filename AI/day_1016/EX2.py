#  V개의 거점과 E개의 도로로 구성되어 있는 지역이 있다. 
# 도로는 거점과 거점 사이에 놓여 있으며, 일방 통행 도로이다. 
# 거점에는 편의상 1번부터 V번까지 번호가 매겨져 있다고 하자.

# 당신은 도로를 따라 기동훈련을 하기 위한 경로를 찾으려고 한다. 
# 기동훈련을 한 후에는 다시 시작점으로 돌아오는 것이 좋기 때문에, 
# 우리는 사이클을 찾기를 원한다. 

# 단, 당신은 기동훈련을 매우 귀찮아하므로, 사이클을 이루는 도로의 
# 길이의 합이 최소가 되도록 찾으려고 한다.

# 도로의 정보가 주어졌을 때, 도로의 길이의 합이 가장 작은 사이클을 찾는 프로그램을 작성하시오. 두 거점을 왕복하
# 는 경우도 사이클에 포함됨에 주의한다.

# 입력
# 첫째 줄에 V와 E가 빈칸을 사이에 두고 주어진다. (2 ≤ 
# V ≤ 400, 0 ≤ E ≤ V(V-1)) 다음 E개의 줄에는 각각 세 개의 정수 a, b, c가 주어진다. 
# a번 거점에서 b번 거점로 가는 거리가 c인 도로가 있다는 의미이다. 
# (a → b임에 주의) 
# 거리는 10,000 이하의 자연수이다. (a, b) 쌍이 같은 
# 도로가 여러 번 주어지지 않는다.
# 출력
# 첫째 줄에 최소 사이클의 도로 길이의 합을 출력한다.
# 훈련 경로를 찾는 것이 불가능한 경우에는 -1을 출력한다.

import sys
input = sys.stdin.readline

INF = int(1e9)

# 입력
V, E = map(int, input().split())
dist = [[INF] * (V + 1) for _ in range(V + 1)]

# 간선 입력
for _ in range(E):
    a, b, c = map(int, input().split())
    dist[a][b] = c  # 방향 그래프임에 주의

# 플로이드 워셜
for k in range(1, V + 1):
    for i in range(1, V + 1):
        for j in range(1, V + 1):
            if dist[i][j] > dist[i][k] + dist[k][j]:
                dist[i][j] = dist[i][k] + dist[k][j]

# 최소 사이클 찾기
ans = INF
for i in range(1, V + 1):
    for j in range(1, V + 1):
        if i != j:
            ans = min(ans, dist[i][j] + dist[j][i])

# 출력
print(ans if ans != INF else -1)
