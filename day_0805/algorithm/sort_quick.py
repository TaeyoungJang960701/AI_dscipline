# 퀵 정렬은 다음과 같은 과정으로 정렬을 진행한다. 정렬은 오름차순으로 진행한다고 가정하자.

# 주어진 배열에서 하나의 요소를 선택하고 이를 pivot(피벗) 으로 삼는다.
# 배열 내부의 모든 값을 검사하면서 피벗 값보다 작은 값들은 왼쪽에, 큰 값들은 오른쪽에 배치한다.
# 이렇게 하면 배열이 두 부분으로 나뉜다. 
# 나뉜 이 두 개의 배열에서 각각 새로운 피벗을 만들어서 두개의 배열로 다시 쪼개어 준다.
# 더 이상 배열을 쪼갤 수 없을 때까지 진행한다.

def quick_sort(a):
    n = len(a)

    if n <= 1:   # 종료 조건(리스트의 크기가 1 이하이다 => 못쪼개지)
        return a

    # 기준 값 피벗(pivot) 설정
    pivot = a[-1] # 별 의미 없음 그냥 관습적으로 마지막 요소를 피벗으로 쓴대
    g1 = []
    g2 = []
    for i in range(0,n-1):
        if a[i] < pivot:
            g1.append(a[i])
        else:
            g2.append(a[i])
    return quick_sort(g1) + [pivot] + quick_sort(g2)


d = [6, 8, 3, 1, 2, 4, 7, 5]
print(quick_sort(d))

print('----------------')
def quick_sort2(a): # 종료 조건 : 정ㄹㄹ 대상이 한 개 이하이면 정렬할 필요 x
    if end - start <= 0:
        return
    
    pivot = a[end]
    i = start
    for j in range(start,end):
            if a[j] <= pivot:
                 a[i],a[j] = a[j], a[i]
                 i += 1
    a[i], a[end] = a[end], a[i]

    quick_sort_sub(a,start,i-1)
    quick_sort_value(a, i + 1,end)

def quick_sort2(a):
    quick_sort_sub(a, 0, len(a), - 1)

d = [6, 8, 3, 1, 2, 4, 7, 5]
quick_sort2(d)
print(d)

