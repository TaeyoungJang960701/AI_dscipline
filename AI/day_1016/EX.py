import heapq
import sys
input = sys.stdin.readline

INF = int(1e9)

# 1) 입력 받기
n, m = map(int, input().split())
graph = [[] for _ in range(n + 1)]
distance = [INF] * (n + 1)

for _ in range(m):
    a, b, c = map(int, input().split())
    # 양방향 연결
    graph[a].append((b, c))
    graph[b].append((a, c))

# 2) 다익스트라 알고리즘
def dijkstra(start):
    q = []
    heapq.heappush(q, (0, start))
    distance[start] = 0

    while q:
        dist, now = heapq.heappop(q)
        if distance[now] < dist:
            continue
        for next_node, cost in graph[now]:
            new_cost = dist + cost
            if new_cost < distance[next_node]:
                distance[next_node] = new_cost
                heapq.heappush(q, (new_cost, next_node))

# 3) 실행
dijkstra(1)

# 4) 결과 출력 (1번 → N번)
print(distance[n])
